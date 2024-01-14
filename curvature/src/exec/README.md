# Execution layer of the Curvature

# Tradeoffs in the `PhysicalExpr` and `PhysicalOperator` trait
We use the push based model to execute the operators and expressions, as we all know, we need an centralized executor to execute it and itself is not executable. However, we design these two traits all have `execution` methods 😂. This maybe pretty confusing...

As we all know, the dynamic polymorphism can be achieved by `Trait Object` and `Enum`. We use trait object in our execution model because it is more flexible, user could add their own implementation easily. If we centralize the execution in the executor, we need to downcast the `Trait Object` to concrete type first, then execute them. It is pretty annoying, because if the user add a new `Expression`/`Operator`, they may forget to add this type in the executor and the program is incorrect. Therefore, we place the `execution` method in the trait and force the user implement it!
# Tittle-tattle

## Do we need intra-pipeline parallelism when we use Morsel-Driven Parallelism?
**NO**