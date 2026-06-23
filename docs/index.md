# OCP Pipeline Documentation

`ovos-ocp-pipeline-plugin` is the OVOS pipeline plugin for specialised
media handling — it classifies media playback and control utterances
and routes them into the OVOS Common Playback (OCP) surface.

Entry points (`opm.pipeline`):

| ID | Class |
|---|---|
| `ovos-ocp-pipeline-plugin` | `ocp_pipeline.opm:OCPPipelineMatcher` |
| `ovos-ocp-pipeline-plugin-legacy` | `ocp_pipeline.opm:MycroftCPSLegacyPipeline` |

## Spec conformance

Language handling is delegated to
[`ovos-spec-tools`](https://github.com/OpenVoiceOS/ovos-spec-tools),
the reference implementation of the
[OpenVoiceOS architecture specifications](https://github.com/OpenVoiceOS/architecture).
The plugin uses `standardize_lang` and `closest_lang` from
`ovos_spec_tools` rather than the deprecated `ovos_utils.lang`
helpers, so language tags are normalised and matched identically
across every conformant component (OVOS-INTENT-2 language matching).

## Tests

```bash
pip install -e .
pytest test/
```
