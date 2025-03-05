use proc_macro::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Fields, parse_macro_input};

#[proc_macro_derive(MetricsSetBuilder)]
pub fn metrics_set_derive(input: TokenStream) -> TokenStream {
    // Parse the input tokens into a syntax tree.
    let input = parse_macro_input!(input as DeriveInput);

    // Used in the quasi-quotation below as `#name`.
    let name = input.ident;

    let insert_recursion = insert_recursion(&input.data);

    let expanded = quote!(
        impl #name {
            fn metrics_set(&self) -> crate::exec::physical_operator::metric::MetricsSet {
                let mut metrics_set = std::collections::HashMap::new();
                #insert_recursion
                crate::exec::physical_operator::metric::MetricsSet{
                    name: stringify!(name),
                    metrics: metrics_set
                }
            }
        }
    );

    expanded.into()
}

fn insert_recursion(data: &Data) -> proc_macro2::TokenStream {
    match *data {
        Data::Struct(ref ds) => match ds.fields {
            Fields::Named(ref fields) => {
                let recurse = fields.named.iter().map(|f| {
                    let name = f.ident.as_ref().unwrap();
                    let f_ty = &f.ty;
                        quote!{
                        metrics_set.insert(stringify!(#f_ty).into(), crate::exec::physical_operator::metric::MetricValue::#f_ty(self.#name.value()));
                    }
                });
                quote! {
                    #(#recurse)*
                }
            }
            _ => panic!("MetricsSet can only be derived for structs with named fields"),
        },
        Data::Enum(_) | Data::Union(_) => {
            panic!("MetricsSet can only be derived for structs with named fields")
        }
    }
}
