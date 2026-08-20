/** The `data-*` boundary: HTML a page author wrote, read once, or refused with a message that says
 * which attribute is missing.
 *
 * Shared rather than copied. It lived inside `frontier.ts` while there was one caller; the third
 * widget made a second one, and the alternative to moving it was a duplicate of the one function
 * whose whole job is to make a boundary failure legible. It imports nothing -- in particular not
 * `mount.js`, which is the cycle that once made the entire bundle throw on load.
 *
 * `label` is the widget's own name, so the reader is told which figure has the bad attribute: a
 * page can carry three mount points and a bare "data-bundle is missing" names none of them.
 */
export function requireAttr(raw: string | undefined, what: string, label: string): string {
  if (raw === undefined || raw === "") throw new Error(`${label}: ${what} is missing`);
  return raw;
}
