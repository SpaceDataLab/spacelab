// Propagate a whole GP catalogue with satellite.js and stream the answers on stdout.
//
// The stream is consumed by `cargo run --example catalogue_sweep -p spacelab-sgp4`, which reads the
// same TLE file and compares. It is a pipe rather than a file because the full sweep is ~9 million
// samples: never worth writing to disk, and the comparison needs no random access.
//
// Two deliberate choices about *what* is propagated:
//
//   * `sgp4(satrec, tsince)` is called with the minute offset directly, not `propagate(satrec,
//     date)`. The date path routes the offset through a JavaScript `Date`, which is quantised to
//     the millisecond -- 7.5 m of along-track at LEO speeds, an artefact of the harness rather than
//     of either implementation. Passing tsince removes the quantisation entirely, so the residual
//     that comes out is the residual between the two transcriptions.
//
//   * Both sides read the *same TLE lines*. Feeding this side a TLE and the other an OMM would
//     conflate the transcription difference with the fixed-width truncation measured separately in
//     crates/spacelab-sgp4/tests/omm_ingest.rs.
//
// Usage: node oracle.cjs SATELLITE_JS CATALOGUE_TLE [FIRST_MIN] [LAST_MIN] [STEP_MIN]

'use strict';

const fs = require('node:fs');

// The build Celestrak serves prints three debug lines per element set from inside twoline2satrec.
// On a 16,000-object catalogue that is 48,000 lines of noise interleaved with the sample stream, so
// it has to go. Silencing the log is the smallest intervention that keeps the fetched file
// byte-identical to its digest -- editing it would break the pin for a cosmetic reason.
console.log = () => {};

const [satellitePath, cataloguePath] = process.argv.slice(2);
const first = Number(process.argv[4] ?? -1440);
const last = Number(process.argv[5] ?? 1440);
const step = Number(process.argv[6] ?? 5);

if (!satellitePath || !cataloguePath) {
  process.stderr.write('usage: node oracle.cjs SATELLITE_JS CATALOGUE_TLE [FIRST] [LAST] [STEP]\n');
  process.exit(2);
}
if (!Number.isFinite(first) || !Number.isFinite(last) || !(step > 0) || last < first) {
  process.stderr.write(`bad offset grid: ${first}..${last} step ${step}\n`);
  process.exit(2);
}

const satellite = require(satellitePath);

// The catalogue is 3LE: a name line then the two element lines. A 2LE file (no name line) is also
// accepted, because Celestrak serves both and the difference is not worth a flag.
const lines = fs
  .readFileSync(cataloguePath, 'utf8')
  .split('\n')
  .map((l) => l.replace(/\r$/, ''))
  .filter((l) => l.trim().length > 0);

const pairs = [];
for (let i = 0; i < lines.length; i += 1) {
  if (lines[i].startsWith('1 ') && i + 1 < lines.length && lines[i + 1].startsWith('2 ')) {
    pairs.push([lines[i], lines[i + 1]]);
    i += 1;
  }
}

const out = [];
let pending = 0;
function emit(line) {
  out.push(line);
  pending += line.length;
  if (pending > (1 << 20)) {
    fs.writeSync(1, out.join(''));
    out.length = 0;
    pending = 0;
  }
}

// The oracle identifies itself in the stream, so the fixture the Rust side writes carries the
// provenance of the numbers in it rather than of whatever is on disk when someone later reads it.
const bytes = fs.readFileSync(satellitePath);
const digest = require('node:crypto').createHash('sha256').update(bytes).digest('hex');
const banner = bytes.toString('utf8', 0, 200).match(/satellite-js v([0-9.]+)/);
emit(`# oracle satellite-js ${banner ? banner[1] : 'unknown'} ${digest}\n`);
emit(`# input ${require('node:path').basename(cataloguePath)} ${pairs.length}\n`);
emit(`# grid ${first} ${last} ${step}\n`);

for (const [line1, line2] of pairs) {
  const satrec = satellite.twoline2satrec(line1, line2);
  // Columns 3-7 of line 1, verbatim: Alpha-5 is decoded on the Rust side, and the key only has to
  // match between the two, not be numeric.
  const key = line1.slice(2, 7).trim();
  if (satrec.error) {
    emit(`I ${key} ${satrec.error}\n`);
    continue;
  }
  emit(`O ${key}\n`);
  for (let t = first; t <= last; t += step) {
    const s = satellite.sgp4(satrec, t);
    // satellite.js signals a failed propagation by returning false, by setting satrec.error, or by
    // handing back a position that is not finite. All three mean the same thing to the comparison
    // and all three have to be checked: the library is not consistent about which it uses.
    if (!s || !s.position || !Number.isFinite(s.position.x)) {
      emit(`E ${t} ${satrec.error || 1}\n`);
      continue;
    }
    const p = s.position;
    const v = s.velocity;
    emit(`S ${t} ${p.x} ${p.y} ${p.z} ${v.x} ${v.y} ${v.z}\n`);
  }
}

if (out.length > 0) {
  fs.writeSync(1, out.join(''));
}
