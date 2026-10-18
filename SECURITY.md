# Security scope

A bounded source review is not a guarantee of zero vulnerabilities. This is local research code, not a hardened multi-user service. Do not run untrusted project code, models or configurations with sensitive credentials or elevated permissions.

## Publication hardening

- New source-only Git history: no model weights, binary checkpoints, run data, original private documents, credentials or personal machine paths are included.
- Replaced pickle-based checkpoint loading with bounded ZIP_STORED archives containing safetensors and schema-validated JSON. Legacy archives, arbitrary object types, unknown archive paths, compressed archives, oversized payloads and unsafe array dtypes are rejected. There is no unsafe fallback. Optimizer/RNG/replay resume still has exact-next-update tests.
- Escaped HTML-significant characters in replay JSON so embedded data cannot terminate its script tag. Replay uses no external scripts or network calls.
- Model loading accepts vetted local directories only, rejects remote/plugin attention config overrides and remote model mappings, uses safetensors, disables remote code and authentication, and never implicitly downloads or calls an inference API.
- Network behavior is limited to the separately invoked fixed-repository/revision artifact download helper, using normal TLS verification. No network listener, shell execution or model-directed tool execution is present in application code.
- CLI output paths are chosen by the local user; do not point them at important directories. Output folders refuse overwrite where applicable. Checkpoints are atomically replaced. Limits constrain normal work but do not sandbox malicious dependencies; use OS isolation for unknown inputs.

## Dependency advisories

The legacy environment used for functional tests includes PyTorch 2.2.0 and Transformers 4.37.2. **That environment has known advisories and is not claimed secure.** No global environment was upgraded. Published dependency recommendations move beyond the reviewed patch levels, but a fresh resolved dependency environment has not yet been fully tested or exhaustively audited.

Reviewed primary advisories include:

- [PyTorch malicious checkpoint memory corruption/RCE, patched 2.10.0](https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p). Merely setting weights_only is not an adequate project mitigation; this source removes torch.load entirely.
- [Transformers malicious attention implementation configuration](https://github.com/advisories/GHSA-29pf-2h5f-8g72), patched 5.3.
- [Transformers save_pretrained chat-template path traversal](https://github.com/advisories/GHSA-xrqw-3rrv-vx5w), patched 5.10.
- [Transformers Trainer RNG-state loading](https://github.com/advisories/GHSA-69w3-r845-3855). This project does not use Trainer resume.

These package advisories do not prove every vulnerable path is reachable here. Keep dependencies current and audit the fully resolved environment before real use. pip-audit, Bandit, Gitleaks and TruffleHog were not installed in the inspected environment; no successful scanner run is claimed. Review used source/history checks, regression tests and official advisories. Modern-runtime compatibility, exhaustive transitive dependency scanning, fuzzing, penetration testing and hostile-model resource isolation remain unverified.

Regression tests cover safe state roundtrip, rejection of pickle/compressed/traversal/dynamic-type payloads, replay escaping, model-load boundaries and private agent-state isolation. Functional tests are not a substitute for ongoing security review. Do not post credentials or private checkpoint contents in public issues.
