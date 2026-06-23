# OCP Pipeline Documentation

`ovos-ocp-pipeline-plugin` is the OVOS pipeline plugin for specialised
media handling — it classifies media playback and control utterances
and routes them into the OVOS Common Playback (OCP) surface.

Entry points (`opm.pipeline`):

| ID | Class |
|---|---|
| `ovos-ocp-pipeline-plugin` | `ocp_pipeline.opm:OCPPipelineMatcher` |
| `ovos-ocp-pipeline-plugin-legacy` | `ocp_pipeline.opm:MycroftCPSLegacyPipeline` |

## In-process MediaProviders (OVOS-OCP-1)

In addition to the bus-broadcast OCP search skills (`ovos.common_play.query`),
the pipeline now dispatches searches to in-process `MediaProvider` plugins
(`opm.media.provider`). This is **additive**: both sources run for the same
query and their results compete in the same normalise/rank/`select_best` flow
by `match_confidence`. Skill-targeted searches (the user named a specific OCP
skill) stay bus-only.

How a query flows through a provider:

- On load, `load_media_providers` discovers installed providers, filtered by
  `is_available()` and the per-provider `enabled` config gate. If the
  `MediaProvider` type or `mediavocab` is missing, or none are enabled,
  `self.media_providers` stays empty and behaviour is identical to the
  bus-only path.
- `ocp_pipeline/bridge.py` is the seam between the two data models. The
  classified `ovos_utils.ocp.MediaType` + text query become a
  `mediavocab.Signals`; each provider is gated by its three-axis `matches()`
  routing test; surviving providers run concurrently via `search_safe`.
- Each returned `mediavocab.Release` is mapped back to the OCP playback result
  dict (`MediaEntry`) — including `mediavocab.taxonomy.PlaybackType` (routing) →
  `ovos_utils.ocp.PlaybackType` (backend selector) — so OCP can keep filtering
  and ranking on its own taxonomy.

This bridges the `mediavocab` catalog model into OCP per the OVOS-OCP-1
architecture spec; `mediavocab>=1.0.0` is now a runtime dependency.

## Tests

```bash
pip install -e .
pytest test/
```
