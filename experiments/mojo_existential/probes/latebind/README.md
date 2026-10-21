Late-bound universe: `lib/factory.mojo` imports `existential_universe`, which
each application provides on its include path. App B adds a trait the library
never saw and gives the library's `Square` a retroactive conformance via
`unsafe_register_as`.

    cd probes/latebind
    mojo run -I ../.. -I lib -I app_a app_a/main.mojo
    mojo run -I ../.. -I lib -I app_b app_b/main.mojo
