---
title: Reachy Motion
emoji: 💬
colorFrom: yellow
colorTo: red
sdk: static
pinned: false
short_description: Talk with Reachy Mini — it moves with what it says.
tags:
 - reachy_mini
 - reachy_mini_python_app
 - voice
 - motion-generation
 - gpt-live
 - elevenlabs
 - taiwanese
---

# Reachy-Motion 💬

**Live voice chat with Reachy Mini — the robot moves with what it says, in real time.**

Every clause of the robot's reply becomes an expressive gesture (head, antennas, body) that lands with the audio,
on top of idle breathing and speech-driven head sway. A small body agent also lets you control it by voice
(mode, volume, "look at me"), lets it see through its camera and look things up on the web.

- **GPT-Live-1** (OpenAI, full duplex) · **ElevenLabs Agents** (your agent, Eleven v4 voices) · **台語** (local models on a Mac)
- Built on Binh Pham's motion-recipe language and [reachy-animation](https://github.com/pham-tuan-binh/reachy-animation)

## Setup after installing

1. Start **Reachy-Motion** from the dashboard and open its settings page (gear icon, or `http://reachy-mini.local:8042`).
2. Under **Keys & voices**, enter your **OpenAI API key** and, for the ElevenLabs mode, your **agent ID**
   (plus an ElevenLabs API key only if the agent is private).
3. Pick a mode and talk. Keys are stored **only on your robot / computer** (`~/.config/reachy_motion/.env`, mode 600);
   they are never part of this Space and never sent anywhere except the provider's API.

Source, docs and the full knowledge base: **https://github.com/dAAAb/Reachy-Motion** (Apache-2.0).
