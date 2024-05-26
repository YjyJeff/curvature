use super::{Operator, Pipeline, PipelineIndex, Sink, Source};
use crate::common::client_context::ClientContext;
use crate::exec::physical_operator::union::Union;
use crate::exec::physical_operator::PhysicalOperator;
use snafu::{ensure, OptionExt, Snafu};
use std::cell::RefCell;
use std::mem::take;
use std::sync::Arc;

#[allow(missing_docs)]
#[derive(Debug, Snafu)]
pub enum PipelineBuilderError {
    #[snafu(display(
        "PipelineBuilder does not support regular/sink operator: `{}`.
         The `{}` operator has {} children. AFAIK, all of the operators,
         except `Union`,`Join` and `TableScan`, should have 1 child",
        op,
        op,
        children_count
    ))]
    UnsupportedOperator {
        op: &'static str,
        children_count: usize,
    },
    #[snafu(display(
        "The pipeline builder encounters TableScan operator,
         but it has children dependencies: {:?}",
        children_indexes,
    ))]
    PipelineWithTableScanHasChildren {
        children_indexes: Vec<PipelineIndex>,
    },
    #[snafu(display(
        "`Pipelines` is already borrowed as mutable, we can 
         not borrow it as mutable anymore. The `PipelineBuilder` has bug 😭"
    ))]
    BorrowChecker,
    #[snafu(display("Pipeline builder try to build a new pipeline, however the source is None."))]
    SourceIsNone,
}

type Result<T> = std::result::Result<T, PipelineBuilderError>;

/// Builder for building pipelines from the tree of the [`PhysicalOperator`]
#[derive(Debug)]
pub(super) struct PipelineBuilder<'p> {
    pipelines: &'p RefCell<Vec<Pipeline<Sink>>>,
    operators: Vec<Operator>,
    source: Option<Source>,

    /// Indexes of the children pipeline
    children_indexes: Vec<PipelineIndex>,
    /// Collection of the left branches of join. Vector here because of join may join on multiple
    /// tables
    join_children_indexes: Vec<PipelineIndex>,
    /// Union Builders created by Union. The union_builder may have its own union_builders because
    /// chain of multiple unions
    union_builders: Vec<Self>,

    /// Client context
    client_ctx: &'p ClientContext,
}

impl<'p> PipelineBuilder<'p> {
    /// Create a new pipeline builder
    pub(super) fn new(
        pipelines: &'p RefCell<Vec<Pipeline<Sink>>>,
        client_ctx: &'p ClientContext,
    ) -> Self {
        Self {
            pipelines,
            operators: vec![],
            source: None,
            children_indexes: vec![],
            join_children_indexes: vec![],
            union_builders: Vec::with_capacity(2),
            client_ctx,
        }
    }

    /// Create union pipeline builder. Used by union
    fn create_union_builder(&mut self) -> &mut Self {
        self.union_builders.push(Self {
            pipelines: self.pipelines,
            operators: vec![],
            source: None,
            children_indexes: vec![],
            join_children_indexes: vec![],
            union_builders: Vec::with_capacity(2),
            client_ctx: self.client_ctx,
        });

        // SAFETY: we push a new builder above
        unsafe { self.union_builders.last_mut().unwrap_unchecked() }
    }

    /// Using DFS to build pipelines.
    ///
    /// Rules:
    ///     1. join's right child depends on both join's left child and the child of itself
    ///     2. sink above union depends on two pipelines that splitted by union
    ///     3. other cases, only depend on the child of itself
    ///
    /// FIXME: when the operator has already been visited. a -> b, c-> b. This happens
    /// in CTE
    pub(super) fn build_pipelines(&mut self, operator: &Arc<dyn PhysicalOperator>) -> Result<()> {
        if self.handle_special_operators(operator)? {
            return Ok(());
        }
        // Normal operators
        let children = operator.children();
        if children.is_empty() {
            // Children is empty, it must be table scan. Set the operator as source
            let source = Source {
                op: Arc::clone(operator),
                global_state: operator.global_source_state(self.client_ctx),
            };
            self.source = Some(source);
            ensure!(
                self.children_indexes.is_empty(),
                PipelineWithTableScanHasChildrenSnafu {
                    children_indexes: self.children_indexes.clone()
                }
            );
        } else {
            ensure!(
                children.len() == 1,
                UnsupportedOperatorSnafu {
                    op: operator.name(),
                    children_count: children.len()
                }
            );

            // build pipeline recursively
            self.build_pipelines(&children[0])?;

            // Till now, we have built pipeline for the child and of the information is stored
            // in self or self.union_builders. Let's handle operator now

            if operator.is_sink() {
                self.handle_sink_operator(operator)?;
            } else {
                // Regular operator, push to operators directly
                let operator = Operator {
                    op: Arc::clone(operator),
                    global_state: operator.global_operator_state(self.client_ctx),
                };
                self.handle_regular_operator(operator);
            }
        }
        Ok(())
    }

    /// FIXME: implement join
    ///
    /// Handle special operators. Returns true if it is a special operator.
    ///
    /// Special operators:
    /// - Union
    /// - Join
    fn handle_special_operators(&mut self, operator: &Arc<dyn PhysicalOperator>) -> Result<bool> {
        let any = operator.as_any();
        if let Some(union_op) = any.downcast_ref::<Union>() {
            // Build pipelines for the left child. The left child is built in a new
            // pipeline builder stored in self.union_builders
            self.create_union_builder()
                .build_pipelines(union_op.left())?;
            // Build pipelines for the right child. The right child is built in a new
            // pipeline builder stored in self.union_builders
            self.create_union_builder()
                .build_pipelines(union_op.right())?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Operator is a sink, we should view the descendant operators of the sink as a pipeline.
    fn handle_sink_operator(&mut self, op: &Arc<dyn PhysicalOperator>) -> Result<()> {
        let sink = Sink {
            op: Arc::clone(op),
            global_state: op.global_sink_state(self.client_ctx),
        };
        if self.union_builders.is_empty() {
            // Do not have union operator between the source and sink
            let pipeline = self.build_pipeline_with_sink(sink)?;
            self.push_pipeline_and_build_dependency(pipeline)?;
        } else {
            // Multiple pipelines share the sink, because of union
            let mut union_builders = take(&mut self.union_builders);

            // Different builders should share the GlobalSinkState
            while let Some(mut builder) = union_builders.pop() {
                if builder.union_builders.is_empty() {
                    let pipeline = builder.build_pipeline_with_sink(sink.clone())?;
                    builder.push_pipeline_and_build_dependency(pipeline)?;
                    self.children_indexes.extend(builder.children_indexes);
                } else {
                    union_builders.extend(builder.union_builders);
                }
            }
        }
        // Sink is the source of the later pipeline
        let source = Source {
            op: Arc::clone(op),
            global_state: op.global_source_state(self.client_ctx),
        };

        self.source = Some(source);

        Ok(())
    }

    fn build_pipeline_with_sink(&mut self, sink: Sink) -> Result<Pipeline<Sink>> {
        let source = self.take_source()?;
        let operators = take(&mut self.operators);
        let children = self.take_children();

        Ok(Pipeline {
            source,
            operators,
            sink,
            children,
        })
    }

    fn take_source(&mut self) -> Result<Source> {
        self.source.take().context(SourceIsNoneSnafu)
    }

    fn take_children(&mut self) -> Vec<PipelineIndex> {
        let mut children = take(&mut self.children_indexes);
        // Handle join
        children.extend(take(&mut self.join_children_indexes));

        children
    }

    /// Push pipeline into pipelines and build dependency between pipelines
    fn push_pipeline_and_build_dependency(&mut self, pipeline: Pipeline<Sink>) -> Result<()> {
        let mut pipelines = self
            .pipelines
            .try_borrow_mut()
            .map_err(|_| PipelineBuilderError::BorrowChecker)?;

        // Get current pipeline index
        let index = pipelines.len();
        // Current pipeline is the child of the later pipeline
        self.children_indexes.push(index);

        // Insert into the pipelines
        pipelines.push(pipeline);

        Ok(())
    }

    fn handle_regular_operator(&mut self, operator: Operator) {
        if self.union_builders.is_empty() {
            self.operators.push(operator)
        } else {
            self.union_builders.iter_mut().for_each(|builder| {
                builder.handle_regular_operator(operator.clone());
            });
        }
    }

    /// Finish build pipeline. Flush the temporal states to [`RootPipeline`]s
    pub(super) fn finish(&mut self) -> Result<Vec<Pipeline<()>>> {
        let mut root_pipelines = Vec::with_capacity(2);
        if self.union_builders.is_empty() {
            // Do not have union operator
            root_pipelines.push(self.build_root_pipeline()?);
        } else {
            // Multiple pipelines share the sink, because of union
            let mut union_builders = take(&mut self.union_builders);

            // Different builders should share the GlobalSinkState
            while let Some(mut builder) = union_builders.pop() {
                if builder.union_builders.is_empty() {
                    root_pipelines.push(builder.build_root_pipeline()?);
                } else {
                    union_builders.extend(builder.union_builders);
                }
            }
        }

        Ok(root_pipelines)
    }

    fn build_root_pipeline(&mut self) -> Result<Pipeline<()>> {
        let source = self.take_source()?;
        let operators = take(&mut self.operators);
        let children = self.take_children();
        Ok(Pipeline {
            source,
            operators,
            sink: (),
            children,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU64;

    use super::*;
    use crate::exec::physical_expr::field_ref::FieldRef;
    use crate::exec::physical_expr::function::aggregate::count::CountStart;
    use crate::exec::physical_expr::function::aggregate::AggregationFunctionExpr;
    use crate::exec::physical_operator::aggregate::simple_aggregate::SimpleAggregate;
    use crate::exec::physical_operator::numbers::Numbers;
    use crate::exec::physical_operator::projection::Projection;
    use data_block::types::LogicalType;

    fn union(
        left: Arc<dyn PhysicalOperator>,
        right: Arc<dyn PhysicalOperator>,
    ) -> Arc<dyn PhysicalOperator> {
        Arc::new(Union::try_new(left, right).unwrap())
    }

    fn projection(input: Arc<dyn PhysicalOperator>) -> Arc<dyn PhysicalOperator> {
        Arc::new(Projection::new(
            input,
            vec![Arc::new(FieldRef::new(
                0,
                LogicalType::UnsignedBigInt,
                "number".to_string(),
            ))],
        ))
    }

    fn numbers() -> Arc<dyn PhysicalOperator> {
        Arc::new(Numbers::new(0, unsafe { NonZeroU64::new_unchecked(4096) }))
    }

    fn aggregate(input: Arc<dyn PhysicalOperator>) -> Arc<dyn PhysicalOperator> {
        Arc::new(
            SimpleAggregate::try_new(
                input,
                vec![AggregationFunctionExpr::try_new(&[], Arc::new(CountStart::new())).unwrap()],
            )
            .unwrap(),
        )
    }

    fn build_pipelines(
        operator: &Arc<dyn PhysicalOperator>,
    ) -> (Vec<Pipeline<()>>, Vec<Pipeline<Sink>>) {
        let pipelines = RefCell::new(vec![]);
        let client_ctx = crate::common::client_context::tests::mock_client_context();
        let mut builder = PipelineBuilder::new(&pipelines, &client_ctx);
        builder.build_pipelines(operator).unwrap();
        let root_pipelines = builder.finish().unwrap();
        (root_pipelines, pipelines.into_inner())
    }

    #[test]
    fn test_table_scan() {
        // Numbers
        let root = numbers();
        let (mut root_pipelines, pipelines) = build_pipelines(&root);
        assert!(pipelines.is_empty());
        assert_eq!(root_pipelines.len(), 1);
        let root_pipeline = root_pipelines.pop().unwrap();
        let expect = expect_test::expect!["Source(Numbers)"];
        expect.assert_eq(&root_pipeline.to_string());
    }

    #[test]
    fn test_projection() {
        //  Projection
        //    Projection
        //      Numbers
        let root = projection(projection(numbers()));
        let (mut root_pipelines, pipelines) = build_pipelines(&root);
        assert!(pipelines.is_empty());
        assert_eq!(root_pipelines.len(), 1);
        let root_pipeline = root_pipelines.pop().unwrap();
        let expect = expect_test::expect!["Source(Numbers) --> Projection --> Projection"];
        expect.assert_eq(&root_pipeline.to_string());
    }

    #[test]
    fn test_aggregate() {
        //  SimpleAggregate
        //    Numbers
        let root = aggregate(numbers());
        let (mut root_pipelines, mut pipelines) = build_pipelines(&root);
        let pipeline = pipelines.pop().unwrap();
        assert!(pipelines.is_empty());
        let expect = expect_test::expect!["Source(Numbers) --> Sink(SimpleAggregate)"];
        expect.assert_eq(&pipeline.to_string());

        let root_pipeline = root_pipelines.pop().unwrap();
        assert!(root_pipelines.is_empty());
        assert_eq!(root_pipeline.children, vec![0]);
        let expect = expect_test::expect!["Source(SimpleAggregate)"];
        expect.assert_eq(&root_pipeline.to_string());
    }

    #[test]
    fn test_aggregate_union() {
        //  Projection
        //    SimpleAggregate
        //      Projection
        //        Union
        //          Numbers
        //          Numbers
        let root = projection(aggregate(projection(union(numbers(), numbers()))));
        let (mut root_pipelines, mut pipelines) = build_pipelines(&root);
        let root_pipeline = root_pipelines.pop().unwrap();

        assert!(root_pipelines.is_empty());
        assert_eq!(root_pipeline.children, vec![0, 1]);
        let expect = expect_test::expect!["Source(SimpleAggregate) --> Projection"];
        expect.assert_eq(&root_pipeline.to_string());

        let p0 = pipelines.pop().unwrap();
        let p1 = pipelines.pop().unwrap();
        assert!(pipelines.is_empty());
        let expect =
            expect_test::expect!["Source(Numbers) --> Projection --> Sink(SimpleAggregate)"];
        expect.assert_eq(&p0.to_string());
        expect.assert_eq(&p1.to_string());

        // Share sink
        assert!(Arc::ptr_eq(&p0.sink.global_state, &p1.sink.global_state));
        assert!(Arc::ptr_eq(&p0.sink.op, &p1.sink.op));
        // Share the regular operator
        assert!(Arc::ptr_eq(&p0.operators[0].op, &p1.operators[0].op));
        assert!(Arc::ptr_eq(
            &p0.operators[0].global_state,
            &p1.operators[0].global_state
        ));
    }

    #[test]
    fn test_aggregate_chain_of_union() {
        //  Projection
        //    SimpleAggregate
        //      Union
        //        Projection
        //          Union
        //            Numbers
        //            Numbers
        //        Numbers
        let root = projection(aggregate(union(
            projection(union(numbers(), numbers())),
            numbers(),
        )));

        let (mut root_pipelines, pipelines) = build_pipelines(&root);
        let root_pipeline = root_pipelines.pop().unwrap();

        assert!(root_pipelines.is_empty());
        assert_eq!(root_pipeline.children, vec![0, 1, 2]);
        let expect = expect_test::expect!["Source(SimpleAggregate) --> Projection"];
        expect.assert_eq(&root_pipeline.to_string());

        assert_eq!(pipelines.len(), 3);
        let p0 = &pipelines[0];
        let p1 = &pipelines[1];
        let p2 = &pipelines[2];

        let expect0 = expect_test::expect![["Source(Numbers) --> Sink(SimpleAggregate)"]];
        let expect1 =
            expect_test::expect!["Source(Numbers) --> Projection --> Sink(SimpleAggregate)"];

        expect0.assert_eq(&p0.to_string());
        expect1.assert_eq(&p1.to_string());
        expect1.assert_eq(&p2.to_string());

        // Share global sink state
        assert!(
            Arc::ptr_eq(&p0.sink.global_state, &p1.sink.global_state)
                && Arc::ptr_eq(&p0.sink.global_state, &p2.sink.global_state)
        );
        assert!(Arc::ptr_eq(&p0.sink.op, &p1.sink.op) && Arc::ptr_eq(&p0.sink.op, &p2.sink.op));

        // Share the regular operator
        assert!(Arc::ptr_eq(&p1.operators[0].op, &p2.operators[0].op));
        assert!(Arc::ptr_eq(
            &p1.operators[0].global_state,
            &p2.operators[0].global_state
        ));
    }

    #[test]
    fn test_union_aggregate() {
        //  Union
        //    Projection
        //      SimpleAggregate
        //        Projection
        //          Union
        //            Numbers
        //            Numbers
        //    SimpleAggregate
        //      Numbers
        let root = union(
            projection(aggregate(projection(union(numbers(), numbers())))),
            aggregate(numbers()),
        );

        let (root_pipelines, pipelines) = build_pipelines(&root);
        assert_eq!(root_pipelines.len(), 2);
        let root_p0 = &root_pipelines[0];
        assert_eq!(root_p0.children, vec![2]);
        let root_p1 = &root_pipelines[1];
        assert_eq!(root_p1.children, vec![0, 1]);

        let expect0 = expect_test::expect!["Source(SimpleAggregate)"];
        let expect1 = expect_test::expect!["Source(SimpleAggregate) --> Projection"];
        expect0.assert_eq(&root_p0.to_string());
        expect1.assert_eq(&root_p1.to_string());

        // pipelines
        assert_eq!(pipelines.len(), 3);
        let p0 = &pipelines[0];
        let p1 = &pipelines[1];
        let p2 = &pipelines[2];

        let expect0 =
            expect_test::expect!["Source(Numbers) --> Projection --> Sink(SimpleAggregate)"];
        expect0.assert_eq(&p0.to_string());
        expect0.assert_eq(&p1.to_string());

        // Share sink
        assert!(Arc::ptr_eq(&p0.sink.global_state, &p1.sink.global_state));
        assert!(Arc::ptr_eq(&p0.sink.op, &p1.sink.op));
        // Share the regular operator
        assert!(Arc::ptr_eq(&p0.operators[0].op, &p1.operators[0].op));
        assert!(Arc::ptr_eq(
            &p0.operators[0].global_state,
            &p1.operators[0].global_state
        ));

        let expect2 = expect_test::expect!["Source(Numbers) --> Sink(SimpleAggregate)"];
        expect2.assert_eq(&p2.to_string())
    }
}
