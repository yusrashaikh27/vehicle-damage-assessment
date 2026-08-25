# Sample photographs for testing the upload form

Eight real photographs, one per damage class plus one spare, copied out of
`dataset/images/test` — the **test** split, so nothing here was seen during
training. That is the honest choice for a demo: showing off on images the model
trained on proves nothing.

All eight are JPEG, 480–640px on the long edge, well under the 10 MB ceiling and
over the 320px minimum, so every one passes upload validation.

| File | What it actually shows | Angle slot | Part shown |
| --- | --- | --- | --- |
| `scratch.jpg` | Dark red car, deep scratch with white abrasion along the rear quarter beside the wheel | Close-up of damage | Rear fender / quarter |
| `scratch_b.jpg` | Bright red car, lighter scuff in the same place | Rear left (3/4) | Rear fender / quarter |
| `dent.jpg` | Red car, front door creased inward below a chrome handle | Left side | Front door |
| `crack.jpg` | Dark green car, bumper corner cracked and torn open below an amber indicator lens | Front right (3/4) | Front bumper |
| `dislocated_part.jpg` | Orange car with the front bumper missing entirely, radiator and crash beam exposed | Front | Front bumper |
| `glass_shatter.jpg` | Cream van, windscreen with a large impact star and radiating cracks | Roof | Windscreen |
| `lamp_broken.jpg` | White pickup, **rear** lamp lens cracked with a piece missing | Rear | Taillamp |
| `tire_flat.jpg` | Red pickup, front tyre deflated and flat-bottomed on the ground | Front left (3/4) | Wheel / tyre |

Two of those entries are worth reading twice. `lamp_broken.jpg` is a **tail**
lamp, not a headlamp, so it belongs on the *Rear* slot with part `Taillamp` — the
pricing table has both, and a taillamp is ₹3,500 against a headlamp's ₹6,500, so
picking the wrong one is a ₹3,000 error. And `dislocated_part.jpg` is a car with
a part *absent* rather than hanging off, which is what that class means after the
merge: it absorbed the export's "no part" and "crash" classes.

## No two of these are the same vehicle

`scratch.jpg` and `scratch_b.jpg` look like a matched pair and are not. They are
two different red cars with similar damage in a similar place — perceptual hash
distance 144 of 256 bits, where 128 is chance, and colour histogram intersection
0.217. The file used to be called `scratch_second_angle.jpg`, which claimed
something untrue, so it was renamed.

This matters less than it looks, and the reason is worth being able to say out
loud: **the merge keys on the component you declare, not on recognising the same
car.** Two photographs both labelled `Rear fender / quarter` merge into one
repair whether or not they show the same vehicle, because the system does not
re-identify damage across views and never claims to. That is exactly why
`core.cost.estimate_combined` takes the **maximum** measured area rather than the
sum: it cannot tell one dent photographed twice from two separate dents, and
understating genuinely separate damage is the safer of the two errors.

Searching the whole test split for a genuine two-angle pair of one vehicle found
none. Everything in the "similar but not identical" band turned out to be
Roboflow rotation augmentations of a single source photograph, which is a
property of the export rather than a gap in the search.

## Details to type in

The form does not check the plate against the photograph, so these are just test
values:

    Full name             Yusra Fathima
    Phone number          +91 98765 43210
    Email                 (leave blank - it is optional)
    Registration number   KA 14 EX 4821
    Make                  Maruti Suzuki
    Model                 Swift
    Year                  2019
    Vehicle segment       Hatchback

Only one of those fields changes the money: **segment**. It sets the part-price
multiplier and the hourly labour rate. Name, plate, make, model and year are
record-keeping and appear on the PDF. Worth knowing before someone asks in the
viva why the estimate ignores the model of the car.

## Three runs worth doing, in this order

The stub detector returns the **same two regions for every photograph**, which is
what makes the comparison below meaningful — the only thing changing between runs
is how many photographs there are and which panel each is labelled with. Numbers
are for Hatchback; a different segment scales them. They were produced by running
`core.pipeline.assess_walkaround` directly against the stub, not estimated.

**Run A — one photograph.** `scratch.jpg` in the *Close-up of damage* slot, part
`Front bumper`. Expect 2 detections and:

    parts ₹0   labour ₹1,125   paint ₹2,500
    subtotal ₹3,625 + GST ₹652 = ₹4,278

**Run B — three photographs, all the same part.** Close-up, Front, and Front left
(3/4) — any three images will do — each with part `Front bumper`. Expect 6
detections across 3 frames, and the identical total:

    subtotal ₹3,625 + GST ₹652 = ₹4,278

Same bill as Run A, from three times the photographs. That is the whole point of
the design: a bumper photographed from three angles is one bumper. The results
page will also carry a note saying the largest single measurement was used rather
than the sum. **This is the single best thing to show an examiner.**

**Run C — three photographs, three different parts.** Same three slots, but parts
`Front bumper`, `Front door`, `Headlamp`. Now the total genuinely rises, because
three different components really are three repairs:

    parts ₹0   labour ₹3,375   paint ₹5,300
    subtotal ₹8,675 + GST ₹1,562 = ₹10,236

B against C is the demonstration: the system does not charge twice for one panel,
but it does charge separately for three. If B had tripled, the merge would be
broken. Note that paint rises by ₹2,800 and not ₹3,000 between them, because a
headlamp is not painted — `PANELS` marks it `paintable=False`.

## Two things that look like bugs and are not

**Every line shows parts ₹0.** The stub's two regions cover 1.00% and 0.96% of
the frame, so both grade as *minor*, and minor damage is repaired rather than
replaced — no part is bought and the bill is labour plus paint only. Correct
behaviour, but it means the demo never exercises the "a panel is bought once"
path, because no part is ever bought. That path is covered by
`tests/cost_baseline.py` instead. The stub's docstring claims its regions are
~3% and ~0.7% and therefore "exercise both severity bands"; that is wrong, and
worth fixing when the stub is next touched.

**A banner says the detections are placeholders.** Until `weights/best.pt` exists
the detector is a stub returning fixed, invented regions that have nothing to do
with what is in the photograph. Upload the shattered-glass image and it will
still report a scratch and a dent. The banner is there so that is never mistaken
for a real measurement — leave it up until training finishes, then re-run the
records from the admin.

## Not in git

This folder's images are excluded from the repository; this README is not. The
images are derived from the Roboflow/CarDD export, which is itself excluded for
size, and they are reproducible from `dataset/` in a few seconds. A teammate who
wants them can pick their own from `dataset/images/test` — but should reject two
things while doing it: images with **black corner padding**, which are rotation
augmentations and look broken in a report, and images carrying **stock-library or
broadcast watermarks**, which are common in the `dislocated_part` class because
it absorbed the web-scraped "crash" images. The first pick for that class here
was an Alamy-watermarked photo and the second was a CBS news screengrab.
