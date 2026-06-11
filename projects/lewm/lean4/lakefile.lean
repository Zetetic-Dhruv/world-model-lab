import Lake
open Lake DSL

package «lewm-validity» where
  leanOptions := #[⟨`autoImplicit, false⟩, ⟨`relaxedAutoImplicit, false⟩]

-- Built against the pinned Mathlib below: run `lake exe cache get` then `lake build`.
require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "8a178386ffc0f5fef0b77738bb5449d50efeea95"

@[default_target]
lean_lib «LewmValidity» where
  -- Build the root module `LewmValidity` and every submodule under `LewmValidity/`,
  -- including the vendored utilities under `LewmValidity/Vendor/`.
  globs := #[Glob.andSubmodules `LewmValidity]
