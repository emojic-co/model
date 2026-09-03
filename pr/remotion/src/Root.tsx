import { Composition } from 'remotion'
import { Scene, sceneDurationInFrames } from './Scene'
import data from './data.json'

const FPS = 30
const WIDTH = 1080
const HEIGHT = 1350

type Entry = { text: string; slug: string }
const DATA = data as Record<string, Entry>

export const RemotionRoot: React.FC = () => {
  return (
    <>
      {Object.values(DATA).map((d) => (
        <Composition
          key={d.slug}
          id={d.slug}
          component={Scene}
          durationInFrames={sceneDurationInFrames(d.text, FPS)}
          fps={FPS}
          width={WIDTH}
          height={HEIGHT}
          defaultProps={{ slug: d.slug }}
        />
      ))}
    </>
  )
}
