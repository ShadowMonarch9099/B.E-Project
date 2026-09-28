// Run at most `max` tasks at once; the rest wait their turn in order.
export function createLimiter(max: number) {
  let active = 0;
  const waiting: (() => void)[] = [];
  const next = () => {
    if (active >= max || waiting.length === 0) return;
    active += 1;
    waiting.shift()!();
  };
  return <T,>(task: () => Promise<T>): Promise<T> =>
    new Promise<T>((resolve, reject) => {
      waiting.push(() =>
        task()
          .then(resolve, reject)
          .finally(() => {
            active -= 1;
            next();
          }),
      );
      next();
    });
}
