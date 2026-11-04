# go-visual

A face made of floating hexagonal tiles. Each tile holds a spinning, morphing mandala, the
colors flow like liquid, and the tiles drift toward and away from you in true perspective,
lit by a small lamp that casts real shadows.

The face is dreamed up from a statistical face model (eigenfaces), so it slowly morphs between
people who don't exist. Turn on the camera and it becomes you.

Plain WebGL2, no build step, no dependencies.

Live: <https://go-visual.vercel.app>

## Run

Any static file server works, for example:

```sh
python3 -m http.server 8765
```

Then open <http://localhost:8765>.

## Controls

Move the mouse to show the controls (bottom right). Moving the pointer also swings the
viewpoint.

| Key   | Action                                      |
|-------|---------------------------------------------|
| `+` `-` | More / bigger hexagons                    |
| `C`   | Camera on / off (face tracking, local only) |
| `T`   | Camera time-lapse (morphs between snapshots) |
| `[` `]` | Shorter / longer time-lapse gap           |
| `E`   | Effects panel (depth, front/back, surge, tilt, perspective, gloss, color speed, breathing) |

Effect settings are saved in the browser.

## Notes

- Camera frames never leave the browser. Face detection runs locally with
  [MediaPipe Tasks Vision](https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector),
  loaded from a CDN.
- `eigenfaces.js` holds a mean face plus 28 principal components (64×64) learned from the
  AT&T / Olivetti research face set. Generated faces are blends, not real people.
