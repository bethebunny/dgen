Probes of the undocumented `__extension` keyword (Mojo 1.1):

    mojo run -I ../.. -I . p16_extension.mojo               # methods + same-module trait conformance
    mojo run -I ../.. -I . p17_extension_existential.mojo   # imported trait: rejected
    mojo run -I ../.. -I . p18_extension_plus_registry.mojo # method-only extension + unsafe_register_as
