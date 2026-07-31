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

## Related projects

- [OpenVoiceOS/ovos-plugin-manager](https://github.com/OpenVoiceOS/ovos-plugin-manager) defines the `ConfidenceMatcherPipeline` base class and the `opm.pipeline` entry-point group this plugin implements.
- [OpenVoiceOS/ovos-workshop](https://github.com/OpenVoiceOS/ovos-workshop) provides `OVOSAbstractApplication`, the bus-app base class this plugin also subclasses.
- [OpenVoiceOS/ovos-bus-client](https://github.com/OpenVoiceOS/ovos-bus-client) provides the OCP message-bus API (`OCPInterface`) used to search and control playback.

## License

Apache-2.0
