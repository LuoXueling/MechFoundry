# Material values

The table below calibrates magnitudes. Every row is a band that real examples of that
material span, not a value to copy, and the answer is a number chosen inside the band
that suits this object or part: its grade, alloy, temper, species, cure, density, moisture, and
the direction it is loaded in all move it. The endpoints are the extremes, so an object
described in neutral terms lands between them, while a qualifier in the request places
it: "soft", "light" or "flexible" belongs near the low end of its row, "hard", "dense"
or "rigid" near the high end -- not in the middle in either case. An object or part between two rows takes a value between them, and a
material the table omits takes its accepted handbook value.

`youngs_modulus` must be expressed in `modulus_unit`. `density` must be expressed in
`density_unit`. The table gives E in pascals, so
convert: 1 MPa = 1e6 Pa, 1 psi = 6895 Pa. Poisson's ratio is dimensionless, is never
converted, and must stay strictly below 0.5, because 0.5 is incompressible and the
solver cannot use it.

The table has no density column: choose density from a standard handbook value, in
kg/m³, appropriate to the same material and condition used for E and ν. Convert kg/m³ to tonne/mm³ with `1 kg/m³ = 1e-12
tonne/mm³`, or to lbf·s²/in⁴ with `1 kg/m³ = 9.357254e-8 lbf·s²/in⁴`.

| material | E (Pa) | ν |
|---|---|---|
| structural steel | 1.9e11 – 2.1e11 | 0.27 – 0.30 |
| stainless steel | 1.9e11 – 2.0e11 | 0.27 – 0.30 |
| cast iron | 8.0e10 – 1.7e11 | 0.21 – 0.29 |
| aluminium alloy | 6.8e10 – 7.3e10 | 0.32 – 0.35 |
| titanium alloy | 1.0e11 – 1.2e11 | 0.32 – 0.34 |
| copper | 1.1e11 – 1.3e11 | 0.33 – 0.36 |
| brass | 9.6e10 – 1.1e11 | 0.33 – 0.36 |
| glass | 6.5e10 – 7.5e10 | 0.21 – 0.25 |
| ceramic, porcelain | 6.0e10 – 1.0e11 | 0.20 – 0.25 |
| stone, marble | 5.0e10 – 7.0e10 | 0.25 – 0.30 |
| concrete | 2.0e10 – 4.0e10 | 0.15 – 0.22 |
| hardwood (walnut, beech, ash), along grain | 1.0e10 – 1.5e10 | 0.30 – 0.40 |
| oak, along grain | 9.0e9 – 1.3e10 | 0.30 – 0.40 |
| pine, along grain | 7.0e9 – 1.2e10 | 0.30 – 0.40 |
| plywood | 5.0e9 – 1.0e10 | 0.25 – 0.35 |
| MDF, particle board | 2.5e9 – 4.5e9 | 0.25 – 0.30 |
| cardboard, paper | 1.0e9 – 5.0e9 | 0.30 – 0.35 |
| bone, cortical | 1.4e10 – 2.0e10 | 0.28 – 0.32 |
| acrylic (PMMA) | 2.4e9 – 3.4e9 | 0.35 – 0.40 |
| polycarbonate | 2.0e9 – 2.6e9 | 0.36 – 0.39 |
| ABS | 1.6e9 – 2.6e9 | 0.33 – 0.37 |
| PLA | 2.7e9 – 3.9e9 | 0.33 – 0.38 |
| nylon | 1.5e9 – 3.0e9 | 0.38 – 0.41 |
| HDPE | 6.0e8 – 1.4e9 | 0.40 – 0.45 |
| PTFE | 4.0e8 – 6.0e8 | 0.45 – 0.47 |
| leather | 5.0e7 – 5.0e8 | 0.35 – 0.45 |
| woven fabric, upholstery textile | 5.0e7 – 5.0e8 | 0.30 – 0.40 |
| felt | 1.0e6 – 1.0e8 | 0.30 – 0.40 |
| rubber, hard | 5.0e7 – 2.0e8 | 0.47 – 0.49 |
| rubber, soft | 1.0e6 – 1.0e7 | 0.48 – 0.49 |
| polyurethane foam | 2.0e4 – 1.0e7 | 0.20 – 0.35 |
| silicone | 1.0e5 – 2.0e6 | 0.45 – 0.49 |
