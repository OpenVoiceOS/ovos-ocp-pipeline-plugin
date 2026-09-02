# OCP Pipeline

The OCP (Open Common Play) pipeline plugin turns media utterances into playback actions in OVOS. Utterances such as "play metallica", "put on some movie", "pause", or "next" pass through this plugin as an intent pipeline stage. On a match, the plugin searches registered media skills over the message bus and drives playback through the OCP API.

The plugin exposes two entry points under the `opm.pipeline` group:

- `ovos-ocp-pipeline-plugin` (`OCPPipelineMatcher`), the main pipeline. It matches `play`, `open`, `media_stop`, `next`, `prev`, `pause`, `resume`, `save_game`, and `load_game` intents at high, medium, or low confidence.
- `ovos-ocp-pipeline-plugin-legacy` (`MycroftCPSLegacyPipeline`), a bridge for older Mycroft CommonPlay skills, using the `play:query` / `play:query.response` / `play:start` handshake.

`OCPPipelineMatcher` also classifies media type, such as music, movie, or podcast. It extracts named entities with `ahocorasick-ner`, using per-language vocabulary files and optional user-supplied entity CSVs (the `entity_csvs` config option). It tracks player state per session through an `OCPPlayerProxy`, kept in sync with `ovos.common_play.status` and `track.state` bus events.

## Architecture

![image](https://github.com/user-attachments/assets/8b6fac59-0e25-4373-ac17-fc5c0b9752fd)

![image](https://github.com/user-attachments/assets/8d7fdf5c-bdd9-4f30-8634-c0a5a6f87359)

## Install

```bash
pip install ovos-ocp-pipeline-plugin
```

## Usage

OVOS core loads this plugin automatically once installed, through the `opm.pipeline` entry-point group. Enable it in `mycroft.conf` under the `intents` section:

```json
{
  "intents": {
    "ovos-ocp-pipeline-plugin": {
      "entity_csvs": []
    }
  }
}
```

The plugin reads the pipeline's `intents` config block, then falls back to a legacy `OCP` config block for backward compatibility.

## Media providers

Besides broadcasting a search over the message bus to OCP skills, the pipeline can query `MediaProvider` plugins in the same process. A provider is a plain catalog: it takes a parsed request and returns candidate releases, with no bus round trip and no skill to run.

Install the whole published set with the `providers` extra, or pick the ones you want one at a time:

```bash
pip install ovos-ocp-pipeline-plugin[providers]
pip install ovos-media-provider-somafm ovos-media-provider-local
```

Nothing else is needed to register them. Each plugin declares an `opm.media.provider` entry point, and the pipeline instantiates every installed provider at startup. A provider that needs credentials or a reachable server, such as Spotify or Music Assistant, loads anyway and answers nothing until you configure it.

Per-provider settings live in `mycroft.conf` under a top-level `media_providers` block, keyed by the provider name as it appears in its entry point. Set `enabled` to `false` there to skip a provider entirely:

```json
{
  "media_providers": {
    "local": {"paths": ["/home/ovos/Music"]},
    "spotify": {"enabled": false}
  }
}
```

To turn the in-process search off altogether and go back to bus skills alone, set `enabled` to `false` on a `media_providers` block inside the pipeline's own config. Be aware that a `media_providers` block in the pipeline config replaces the top-level one as the source of per-provider settings, so keep provider settings in one place or the other:

```json
{
  "intents": {
    "ovos-ocp-pipeline-plugin": {
      "media_providers": {"enabled": false}
    }
  }
}
```

Both searches run for every query that does not name a specific skill, and their results are pooled and ranked together. A provider answer never displaces a skill answer: the only thing dropped is a provider entry pointing at a URI a skill already offered.

## Related projects

- [OpenVoiceOS/ovos-plugin-manager](https://github.com/OpenVoiceOS/ovos-plugin-manager) defines the `ConfidenceMatcherPipeline` base class and the `opm.pipeline` entry-point group this plugin implements.
- [OpenVoiceOS/ovos-workshop](https://github.com/OpenVoiceOS/ovos-workshop) provides `OVOSAbstractApplication`, the bus-app base class this plugin also subclasses.
- [OpenVoiceOS/ovos-bus-client](https://github.com/OpenVoiceOS/ovos-bus-client) provides the OCP message-bus API (`OCPInterface`) used to search and control playback.

## License

Apache-2.0
