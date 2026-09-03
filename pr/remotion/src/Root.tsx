import { Composition } from 'remotion'
import { Scene, START_DELAY_S, CPS, HOLD_S } from './Scene'
import data from './data.json'

const FPS = 30
const WIDTH = 1080
const HEIGHT = 1920

type Entry = { text: string; slug: string }
const DATA = data as Record<string, Entry>

export const RemotionRoot: React.FC = () => {
  return (
    <>
      {Object.values(DATA).map((d) => {
        const durationInFrames = Math.ceil(
          (START_DELAY_S + d.text.length / CPS + HOLD_S) * FPS,
        )
        return (
          <Composition
            key={d.slug}
            id={d.slug}
            component={Scene}
            durationInFrames={durationInFrames}
            fps={FPS}
            width={WIDTH}
            height={HEIGHT}
            defaultProps={{ slug: d.slug }}
          />
        )
      })}
    </>
  )
}
