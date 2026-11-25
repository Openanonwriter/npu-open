# Working rules for this repository

These rules come from the project owner. They apply to every change.

## Language

- **Prefer Rust** for new programs on the Linux side: servers, clients, tools, tests.
- Standard library only unless a crate is clearly worth it; say why in the docs when one is added.
- Exceptions that stay as they are:
  - DSP-side code is C and assembly, built with `clang --target=hexagon` (there is no usable Rust
    target for the Hexagon DSP).
  - Existing C clients and Python scripts are kept working; port them when they are next touched.

## Documentation: always, and meticulously

Every piece of work updates the documentation in the same change, without being asked.

| File | Update when |
|---|---|
| `docs/TECHNICAL.md` | Any new fact: a layout, a return code, a timing, a limit, a quirk, a protocol detail |
| `docs/JOURNAL.md` | Every session: what was tried, in order, including what failed and why |
| `docs/SETUP.md` | Anything a new user must install, build, configure or run differently |
| `README.md` | Status, layout or headline numbers change |
| Source comments | Why the code is the way it is, and where each fact came from |

- Record failures, dead ends and mistakes, not only what worked.
- Record exact values: error codes, sizes, timings with units, versions, commit hashes.
- State how each fact is known: open source (name the file), public manual, or experiment.
- State what is assumed or unverified as such.
- Keep docs free of home paths, usernames, email addresses and captured data.

## Clean-room rules

- No Qualcomm SDK, headers, or licence click-throughs.
- Do not use Qualcomm's code and do not disassemble or otherwise reverse engineer Qualcomm
  binaries, including reading their ELF metadata. Ask the owner first if it seems necessary.
- Qualcomm's firmware and shell are run unmodified, from the owner's own device. They are never
  committed (`.gitignore` covers them).
- Allowed sources: open-licensed code (LLVM, llama.cpp, qualcomm/fastrpc, hexagonrpc), Qualcomm's
  public documents, and black-box experiments on our own hardware.

## Safety on the DSP

- Never stop `remoteproc0` (audio DSP: charging, sensors).
- Explain the risk before any power, clock or firmware experiment, and watch temperatures.
- Power the accelerator before locking it; an unpowered lock resets the DSP.
- After a reboot or a DSP crash, run `./cdsp78.sh` before any test.
- Do not SIGKILL a process with a FastRPC call in flight.

## Results

- Verify accelerator results against a reference before claiming they work, but report the
  accelerator's result and time; do not present CPU comparisons as the headline.
- Never commit firmware, shells, datasets, trained models, or captured traffic.
