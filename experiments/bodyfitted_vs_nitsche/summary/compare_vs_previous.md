# Verification against the reference run

391 of 400 entries match (status, and metrics within 1e-05 absolute)

| session | entry | path (new) | path (reference) | status same | max abs metric change | matches |
|---|---|---|---|---|---|---|
| adjustable-wrench | d7 dofs | gpu jaxamg | gpu jaxamg | yes | 5.9e-08 | yes |
| adjustable-wrench | d7 graded | gpu jaxamg | gpu jaxamg | yes | 1.4e-07 | yes |
| adjustable-wrench | d7 nitsche | jaxamg ls 1 | jaxamg ls 1 | yes | - | yes |
| adjustable-wrench | d7 surface | gpu jaxamg | gpu jaxamg | yes | 6.8e-08 | yes |
| adjustable-wrench | d8 dofs | gpu jaxamg | gpu jaxamg | yes | 1.3e-08 | yes |
| adjustable-wrench | d8 graded | gpu jaxamg | gpu jaxamg | yes | 9.1e-08 | yes |
| adjustable-wrench | d8 nitsche | jaxamg ls 1 | jaxamg ls 1 | yes | - | yes |
| adjustable-wrench | d8 surface | gpu jaxamg | gpu jaxamg | yes | 1.8e-06 | yes |
| bar-stool | d7 dofs | gpu jaxamg | gpu jaxamg | yes | 2.0e-09 | yes |
| bar-stool | d7 graded | gpu jaxamg_4000 | gpu jaxamg | yes | 1.0e-08 | yes |
| bar-stool | d7 nitsche | jaxamg ls 1 | jaxamg | yes | - | yes |
| bar-stool | d7 surface | gpu jaxamg_4000 | gpu jaxamg | yes | 2.5e-09 | yes |
| bar-stool | d8 dofs | gpu pyamgx | gpu jaxamg | yes | 1.8e-07 | yes |
| bar-stool | d8 graded | gpu pyamgx | gpu jaxamg | yes | 9.5e-09 | yes |
| bar-stool | d8 nitsche | pyamgx ls 1 | pyamgx ls 1 | yes | - | yes |
| bar-stool | d8 surface | gpu pyamgx | gpu jaxamg | yes | 9.1e-08 | yes |
| car-brake-disc | d7 dofs | failed: generated surface self-intersects (TetGen -d); not repaired | failed: generated surface self-intersects (TetGen -d); not repaired | yes | - | yes |
| car-brake-disc | d7 graded | failed: generated surface self-intersects (TetGen -d); not repaired | failed: generated surface self-intersects (TetGen -d); not repaired | yes | - | yes |
| car-brake-disc | d7 nitsche | jaxamg ls 1 | jaxamg | yes | - | yes |
| car-brake-disc | d7 surface | failed: generated surface self-intersects (TetGen -d); not repaired | failed: generated surface self-intersects (TetGen -d); not repaired | yes | - | yes |
| car-brake-disc | d8 dofs | failed: generated surface self-intersects (TetGen -d); not repaired | failed: generated surface self-intersects (TetGen -d); not repaired | yes | - | yes |
| car-brake-disc | d8 graded | failed: generated surface self-intersects (TetGen -d); not repaired | failed: generated surface self-intersects (TetGen -d); not repaired | yes | - | yes |
| car-brake-disc | d8 nitsche | jaxamg ls 1 | jaxamg | yes | - | yes |
| car-brake-disc | d8 surface | failed: generated surface self-intersects (TetGen -d); not repaired | failed: generated surface self-intersects (TetGen -d); not repaired | yes | - | yes |
| car-suspension-control-arm-with-rubber-bushings | d7 dofs | gpu jaxamg | gpu jaxamg | yes | 2.7e-08 | yes |
| car-suspension-control-arm-with-rubber-bushings | d7 graded | gpu jaxamg | gpu jaxamg | yes | 2.4e-08 | yes |
| car-suspension-control-arm-with-rubber-bushings | d7 nitsche | jaxamg ls 1 | jaxamg | yes | - | yes |
| car-suspension-control-arm-with-rubber-bushings | d7 surface | gpu jaxamg | gpu jaxamg | yes | 2.6e-08 | yes |
| car-suspension-control-arm-with-rubber-bushings | d8 dofs | gpu jaxamg | gpu jaxamg | yes | 1.3e-08 | yes |
| car-suspension-control-arm-with-rubber-bushings | d8 graded | gpu jaxamg | gpu jaxamg | yes | 2.2e-08 | yes |
| car-suspension-control-arm-with-rubber-bushings | d8 nitsche | jaxamg ls 1 | jaxamg | yes | - | yes |
| car-suspension-control-arm-with-rubber-bushings | d8 surface | gpu jaxamg | gpu jaxamg | yes | 2.8e-08 | yes |
| car-wheel-with-its-rubber-tyre | d7 dofs | gpu jaxamg | gpu jaxamg | yes | 5.1e-10 | yes |
| car-wheel-with-its-rubber-tyre | d7 graded | gpu jaxamg | gpu jaxamg | yes | 3.5e-10 | yes |
| car-wheel-with-its-rubber-tyre | d7 nitsche | jaxamg ls 1 | jaxamg | yes | - | yes |
| car-wheel-with-its-rubber-tyre | d7 surface | gpu jaxamg | gpu jaxamg | yes | 4.2e-10 | yes |
| car-wheel-with-its-rubber-tyre | d8 dofs | cpu (modified Newton) | cpu (modified Newton) | yes | 2.9e-09 | yes |
| car-wheel-with-its-rubber-tyre | d8 graded | cpu (modified Newton) | cpu (modified Newton) | yes | 2.7e-09 | yes |
| car-wheel-with-its-rubber-tyre | d8 nitsche | jaxamg ls 1 | jaxamg | yes | - | yes |
| car-wheel-with-its-rubber-tyre | d8 surface | cpu (modified Newton) | cpu (modified Newton) | yes | 1.7e-09 | yes |
| claw-hammer | d7 dofs | gpu jaxamg | gpu jaxamg | yes | 4.7e-09 | yes |
| claw-hammer | d7 graded | gpu jaxamg | gpu jaxamg | yes | 2.4e-09 | yes |
| claw-hammer | d7 nitsche | jaxamg ls 1 | jaxamg ls 1 | yes | - | yes |
| claw-hammer | d7 surface | gpu jaxamg | gpu jaxamg | yes | 2.2e-09 | yes |
| claw-hammer | d8 dofs | gpu jaxamg | gpu jaxamg | yes | 6.9e-08 | yes |
| claw-hammer | d8 graded | gpu jaxamg | gpu jaxamg | yes | 5.4e-08 | yes |
| claw-hammer | d8 nitsche | jaxamg ls 1 | jaxamg ls 1 | yes | - | yes |
| claw-hammer | d8 surface | gpu jaxamg | gpu jaxamg | yes | 5.8e-08 | yes |
| coffee-mug | d7 dofs | gpu jaxamg | gpu jaxamg | yes | 3.6e-10 | yes |
| coffee-mug | d7 graded | gpu jaxamg | gpu jaxamg | yes | 8.2e-10 | yes |
| coffee-mug | d7 nitsche | jaxamg ls 1 | jaxamg | yes | - | yes |
| coffee-mug | d7 surface | gpu jaxamg | gpu jaxamg | yes | 6.9e-10 | yes |
| coffee-mug | d8 dofs | gpu jaxamg | gpu jaxamg | yes | 3.7e-10 | yes |
| coffee-mug | d8 graded | gpu jaxamg | gpu jaxamg | yes | 3.8e-10 | yes |
| coffee-mug | d8 nitsche | jaxamg ls 1 | jaxamg | yes | - | yes |
| coffee-mug | d8 surface | cpu (modified Newton) | cpu (modified Newton) | yes | 2.7e-10 | yes |
| computer-mouse | d7 dofs | gpu jaxamg | gpu jaxamg | yes | 1.7e-09 | yes |
| computer-mouse | d7 graded | gpu jaxamg | gpu jaxamg | yes | 6.8e-09 | yes |
| computer-mouse | d7 nitsche | jaxamg ls 1 | jaxamg | yes | - | yes |
| computer-mouse | d7 surface | gpu jaxamg | gpu jaxamg | yes | 3.3e-09 | yes |
| computer-mouse | d8 dofs | gpu jaxamg | gpu jaxamg | yes | 5.8e-10 | yes |
| computer-mouse | d8 graded | gpu jaxamg | gpu jaxamg | yes | 9.0e-10 | yes |
| computer-mouse | d8 nitsche | jaxamg ls 1 | jaxamg | yes | - | yes |
| computer-mouse | d8 surface | cpu (modified Newton) | cpu (modified Newton) | yes | 1.9e-11 | yes |
| crankshaft | d7 dofs | gpu jaxamg | gpu jaxamg | yes | 2.8e-08 | yes |
| crankshaft | d7 graded | gpu jaxamg | gpu jaxamg | yes | 2.9e-08 | yes |
| crankshaft | d7 nitsche | jaxamg ls 1 | jaxamg ls 1 | yes | - | yes |
| crankshaft | d7 surface | gpu jaxamg | gpu jaxamg | yes | 3.3e-08 | yes |
| crankshaft | d8 dofs | gpu jaxamg | gpu jaxamg | yes | 2.1e-09 | yes |
| crankshaft | d8 graded | gpu jaxamg | gpu jaxamg | yes | 3.5e-09 | yes |
| crankshaft | d8 nitsche | jaxamg ls 1 | jaxamg ls 1 | yes | - | yes |
| crankshaft | d8 surface | gpu jaxamg | gpu jaxamg | yes | 2.0e-09 | yes |
| crocodile | d7 dofs | gpu jaxamg | gpu jaxamg | yes | 1.8e-08 | yes |
| crocodile | d7 graded | gpu jaxamg | gpu jaxamg | yes | 4.1e-09 | yes |
| crocodile | d7 nitsche | jaxamg ls 1 | jaxamg ls 1 | yes | - | yes |
| crocodile | d7 surface | gpu jaxamg | gpu jaxamg | yes | 1.9e-08 | yes |
| crocodile | d8 dofs | gpu jaxamg_4000 | gpu jaxamg | yes | 1.7e-09 | yes |
| crocodile | d8 graded | gpu jaxamg_4000 | gpu jaxamg | yes | 1.8e-09 | yes |
| crocodile | d8 nitsche | jaxamg ls 1 | jaxamg ls 1 | yes | - | yes |
| crocodile | d8 surface | gpu pyamgx | gpu jaxamg | yes | 4.8e-09 | yes |
| dental-crown | d7 dofs | gpu jaxamg | gpu jaxamg | yes | 2.0e-03 | NO |
| dental-crown | d7 graded | gpu jaxamg | gpu jaxamg | yes | 1.4e-09 | yes |
| dental-crown | d7 nitsche | jaxamg ls 1 | jaxamg | yes | - | yes |
| dental-crown | d7 surface | cpu (modified Newton) | cpu (modified Newton) | yes | 1.2e-10 | yes |
| dental-crown | d8 dofs | cpu (modified Newton) | cpu (modified Newton) | yes | 1.9e-10 | yes |
| dental-crown | d8 graded | cpu (modified Newton) | cpu (modified Newton) | yes | 8.1e-11 | yes |
| dental-crown | d8 nitsche | jaxamg ls 1 | jaxamg | yes | - | yes |
| dental-crown | d8 surface | cpu (modified Newton) | cpu (modified Newton) | yes | 1.3e-10 | yes |
| electric-toothbrush | d7 dofs | gpu jaxamg | gpu jaxamg | yes | 1.5e-07 | yes |
| electric-toothbrush | d7 graded | gpu jaxamg | gpu jaxamg | yes | 1.5e-07 | yes |
| electric-toothbrush | d7 nitsche | jaxamg ls 1 | jaxamg ls 1 | yes | - | yes |
| electric-toothbrush | d7 surface | gpu jaxamg | gpu jaxamg | yes | 9.6e-08 | yes |
| electric-toothbrush | d8 dofs | gpu jaxamg | gpu jaxamg | yes | 7.0e-08 | yes |
| electric-toothbrush | d8 graded | gpu jaxamg | gpu jaxamg | yes | 1.3e-07 | yes |
| electric-toothbrush | d8 nitsche | jaxamg ls 1 | jaxamg ls 1 | yes | - | yes |
| electric-toothbrush | d8 surface | gpu jaxamg | gpu jaxamg | yes | 1.2e-07 | yes |
| elephant | d7 dofs | gpu jaxamg | gpu jaxamg | yes | 4.6e-10 | yes |
| elephant | d7 graded | gpu jaxamg | gpu jaxamg | yes | 3.1e-10 | yes |
| elephant | d7 nitsche | jaxamg ls 1 | jaxamg | yes | - | yes |
| elephant | d7 surface | gpu jaxamg | gpu jaxamg | yes | 7.5e-10 | yes |
| elephant | d8 dofs | gpu jaxamg | gpu jaxamg | yes | 2.4e-09 | yes |
| elephant | d8 graded | gpu jaxamg | gpu jaxamg | yes | 1.8e-09 | yes |
| elephant | d8 nitsche | jaxamg ls 1 | jaxamg | yes | - | yes |
| elephant | d8 surface | gpu jaxamg | gpu jaxamg | yes | 2.0e-09 | yes |
| giraffe | d7 dofs | gpu jaxamg | gpu jaxamg | yes | 9.6e-10 | yes |
| giraffe | d7 graded | gpu jaxamg | gpu jaxamg | yes | 2.3e-09 | yes |
| giraffe | d7 nitsche | jaxamg ls 1 | jaxamg ls 1 | yes | - | yes |
| giraffe | d7 surface | gpu jaxamg | gpu jaxamg | yes | 1.6e-09 | yes |
| giraffe | d8 dofs | gpu pyamgx | gpu jaxamg | yes | 2.4e-06 | yes |
| giraffe | d8 graded | gpu jaxamg_4000 | gpu jaxamg | yes | 1.3e-08 | yes |
| giraffe | d8 nitsche | jaxamg ls 1 | jaxamg ls 1 | yes | - | yes |
| giraffe | d8 surface | gpu pyamgx | gpu jaxamg | yes | 9.8e-08 | yes |
| glass-top-dining-table | d7 dofs | gpu jaxamg | gpu jaxamg | yes | 8.6e-11 | yes |
| glass-top-dining-table | d7 graded | gpu jaxamg | gpu jaxamg | yes | 3.3e-11 | yes |
| glass-top-dining-table | d7 nitsche | jaxamg ls 1 | jaxamg | yes | - | yes |
| glass-top-dining-table | d7 surface | gpu jaxamg | gpu jaxamg | yes | 1.7e-10 | yes |
| glass-top-dining-table | d8 dofs | gpu pyamgx | gpu jaxamg | yes | 2.0e-09 | yes |
| glass-top-dining-table | d8 graded | gpu pyamgx | gpu jaxamg | yes | 2.2e-09 | yes |
| glass-top-dining-table | d8 nitsche | jaxamg ls 1 | jaxamg | yes | - | yes |
| glass-top-dining-table | d8 surface | gpu jaxamg_4000 | gpu jaxamg | yes | 2.0e-09 | yes |
| hamburger | d7 dofs | gpu jaxamg | gpu jaxamg | yes | 1.0e-10 | yes |
| hamburger | d7 graded | gpu jaxamg | gpu jaxamg | yes | 4.9e-11 | yes |
| hamburger | d7 nitsche | jaxamg ls 5 | jaxamg | yes | - | yes |
| hamburger | d7 surface | gpu jaxamg | gpu jaxamg | yes | 2.7e-10 | yes |
| hamburger | d8 dofs | gpu jaxamg | gpu jaxamg | yes | 7.0e-10 | yes |
| hamburger | d8 graded | gpu jaxamg | gpu jaxamg | yes | 1.7e-10 | yes |
| hamburger | d8 nitsche | jaxamg ls 5 | pyamgx ls 5 | yes | - | yes |
| hamburger | d8 surface | lean-gpu | lean-gpu | yes | 1.2e-12 | yes |
| himeji-castle | d7 dofs | failed: generated surface self-intersects (TetGen -d); not repaired | failed: generated surface self-intersects (TetGen -d); not repaired | yes | - | yes |
| himeji-castle | d7 graded | failed: generated surface self-intersects (TetGen -d); not repaired | failed: generated surface self-intersects (TetGen -d); not repaired | yes | - | yes |
| himeji-castle | d7 nitsche | jaxamg ls 1 | jaxamg | yes | - | yes |
| himeji-castle | d7 surface | failed: generated surface self-intersects (TetGen -d); not repaired | failed: generated surface self-intersects (TetGen -d); not repaired | yes | - | yes |
| himeji-castle | d8 dofs | failed: generated surface self-intersects (TetGen -d); not repaired | failed: generated surface self-intersects (TetGen -d); not repaired | yes | - | yes |
| himeji-castle | d8 graded | failed: generated surface self-intersects (TetGen -d); not repaired | failed: generated surface self-intersects (TetGen -d); not repaired | yes | - | yes |
| himeji-castle | d8 nitsche | jaxamg ls 1 | jaxamg | yes | - | yes |
| himeji-castle | d8 surface | failed: generated surface self-intersects (TetGen -d); not repaired | failed: generated surface self-intersects (TetGen -d); not repaired | yes | - | yes |
| horse | d7 dofs | gpu jaxamg | gpu jaxamg | yes | 9.0e-09 | yes |
| horse | d7 graded | gpu jaxamg | gpu jaxamg | yes | 1.7e-08 | yes |
| horse | d7 nitsche | jaxamg ls 1 | jaxamg | yes | - | yes |
| horse | d7 surface | gpu jaxamg | gpu jaxamg | yes | 1.9e-08 | yes |
| horse | d8 dofs | gpu pyamgx | gpu jaxamg | yes | 1.6e-07 | yes |
| horse | d8 graded | gpu pyamgx | gpu jaxamg | yes | 1.4e-07 | yes |
| horse | d8 nitsche | jaxamg_4000 ls 1 | pyamgx ls 1 | yes | - | yes |
| horse | d8 surface | gpu pyamgx | gpu jaxamg | yes | 4.5e-07 | yes |
| human-femur | d7 dofs | gpu jaxamg | gpu jaxamg | yes | 8.4e-08 | yes |
| human-femur | d7 graded | gpu jaxamg | gpu jaxamg | yes | 6.4e-07 | yes |
| human-femur | d7 nitsche | jaxamg ls 1 | jaxamg ls 1 | yes | - | yes |
| human-femur | d7 surface | gpu jaxamg | gpu jaxamg | yes | 3.2e-08 | yes |
| human-femur | d8 dofs | gpu jaxamg | gpu jaxamg | yes | 8.9e-08 | yes |
| human-femur | d8 graded | gpu jaxamg | gpu jaxamg | yes | 5.6e-08 | yes |
| human-femur | d8 nitsche | jaxamg ls 1 | jaxamg ls 1 | yes | - | yes |
| human-femur | d8 surface | gpu jaxamg | gpu jaxamg | yes | 8.0e-08 | yes |
| igloo | d7 dofs | gpu jaxamg | gpu jaxamg | yes | 9.3e-09 | yes |
| igloo | d7 graded | gpu jaxamg | gpu jaxamg | yes | 1.8e-09 | yes |
| igloo | d7 nitsche | jaxamg ls 1 | jaxamg | yes | - | yes |
| igloo | d7 surface | gpu jaxamg | gpu jaxamg | yes | 7.0e-09 | yes |
| igloo | d8 dofs | gpu jaxamg | gpu jaxamg | yes | 7.8e-09 | yes |
| igloo | d8 graded | gpu jaxamg | gpu jaxamg | yes | 2.4e-09 | yes |
| igloo | d8 nitsche | jaxamg ls 1 | jaxamg | yes | - | yes |
| igloo | d8 surface | cpu (modified Newton) | cpu (modified Newton) | yes | 6.3e-11 | yes |
| industrial-robot-arm | d7 dofs | gpu jaxamg | gpu jaxamg | yes | 5.6e-10 | yes |
| industrial-robot-arm | d7 graded | gpu jaxamg | gpu jaxamg | yes | 4.8e-10 | yes |
| industrial-robot-arm | d7 nitsche | jaxamg ls 1 | jaxamg ls 1 | yes | - | yes |
| industrial-robot-arm | d7 surface | gpu jaxamg | gpu jaxamg | yes | 3.5e-10 | yes |
| industrial-robot-arm | d8 dofs | gpu jaxamg | gpu jaxamg | yes | 3.1e-09 | yes |
| industrial-robot-arm | d8 graded | gpu jaxamg | gpu jaxamg | yes | 5.7e-09 | yes |
| industrial-robot-arm | d8 nitsche | jaxamg ls 1 | jaxamg ls 1 | yes | - | yes |
| industrial-robot-arm | d8 surface | gpu jaxamg | gpu jaxamg | yes | 5.7e-09 | yes |
| jackhammer | d7 dofs | cpu (full Newton) | cpu (full Newton) | yes | 9.8e-08 | yes |
| jackhammer | d7 graded | cpu (full Newton) | cpu (full Newton) | yes | 4.5e-07 | yes |
| jackhammer | d7 nitsche | jaxamg ls 1 | jaxamg ls 1 | yes | - | yes |
| jackhammer | d7 surface | cpu (full Newton) | gpu amgx | yes | 2.4e-06 | yes |
| jackhammer | d8 dofs | cpu (full Newton) | gpu amgx | no | - | NO |
| jackhammer | d8 graded | cpu (full Newton) | cpu (full Newton) | no | - | NO |
| jackhammer | d8 nitsche | pyamgx ls 1 | failed | no | - | NO |
| jackhammer | d8 surface | cpu (full Newton) | cpu (full Newton) | no | - | NO |
| jelly-dessert | d7 dofs | gpu jaxamg | gpu jaxamg | yes | 3.0e-10 | yes |
| jelly-dessert | d7 graded | gpu jaxamg | gpu jaxamg | yes | 1.5e-09 | yes |
| jelly-dessert | d7 nitsche | jaxamg ls 1 | jaxamg ls 1 | yes | - | yes |
| jelly-dessert | d7 surface | gpu jaxamg | gpu jaxamg | yes | 8.2e-11 | yes |
| jelly-dessert | d8 dofs | gpu jaxamg | gpu jaxamg | yes | 4.2e-09 | yes |
| jelly-dessert | d8 graded | gpu jaxamg | gpu jaxamg | yes | 2.7e-09 | yes |
| jelly-dessert | d8 nitsche | jaxamg ls 1 | jaxamg ls 1 | yes | - | yes |
| jelly-dessert | d8 surface | lean-gpu | lean-gpu | yes | 1.6e-11 | yes |
| knee-implant | d7 dofs | gpu jaxamg | gpu jaxamg | yes | 6.5e-09 | yes |
| knee-implant | d7 graded | gpu jaxamg | gpu jaxamg | yes | 1.6e-09 | yes |
| knee-implant | d7 nitsche | jaxamg ls 1 | jaxamg | yes | - | yes |
| knee-implant | d7 surface | gpu jaxamg | gpu jaxamg | yes | 8.8e-10 | yes |
| knee-implant | d8 dofs | cpu (modified Newton) | cpu (modified Newton) | yes | 1.1e-09 | yes |
| knee-implant | d8 graded | gpu jaxamg | gpu jaxamg | yes | 1.1e-09 | yes |
| knee-implant | d8 nitsche | jaxamg ls 1 | jaxamg | yes | - | yes |
| knee-implant | d8 surface | cpu (modified Newton) | cpu (modified Newton) | yes | 1.1e-09 | yes |
| layered-birthday-cake | d7 dofs | gpu jaxamg | gpu jaxamg | yes | 1.2e-08 | yes |
| layered-birthday-cake | d7 graded | gpu jaxamg | gpu jaxamg | yes | 7.3e-09 | yes |
| layered-birthday-cake | d7 nitsche | jaxamg ls 1 | jaxamg ls 1 | yes | - | yes |
| layered-birthday-cake | d7 surface | gpu jaxamg | gpu jaxamg | yes | 2.5e-09 | yes |
| layered-birthday-cake | d8 dofs | gpu jaxamg | gpu jaxamg | yes | 4.5e-09 | yes |
| layered-birthday-cake | d8 graded | gpu jaxamg | gpu jaxamg | yes | 1.1e-09 | yes |
| layered-birthday-cake | d8 nitsche | jaxamg ls 1 | jaxamg ls 1 | yes | - | yes |
| layered-birthday-cake | d8 surface | lean-gpu | lean-gpu | yes | 3.0e-13 | yes |
| leaning-tower-of-pisa | d7 dofs | gpu jaxamg_4000 | gpu jaxamg | yes | 1.5e-10 | yes |
| leaning-tower-of-pisa | d7 graded | gpu jaxamg_4000 | gpu jaxamg | yes | 5.5e-10 | yes |
| leaning-tower-of-pisa | d7 nitsche | jaxamg ls 1 | jaxamg | yes | - | yes |
| leaning-tower-of-pisa | d7 surface | gpu pyamgx | gpu jaxamg | yes | 3.5e-07 | yes |
| leaning-tower-of-pisa | d8 dofs | gpu pyamgx | gpu jaxamg | yes | 2.6e-07 | yes |
| leaning-tower-of-pisa | d8 graded | gpu jaxamg_4000 | gpu jaxamg | yes | 9.3e-08 | yes |
| leaning-tower-of-pisa | d8 nitsche | jaxamg ls 1 | jaxamg | yes | - | yes |
| leaning-tower-of-pisa | d8 surface | gpu pyamgx | gpu jaxamg | yes | 1.2e-07 | yes |
| lighthouse | d7 dofs | gpu jaxamg | gpu jaxamg | yes | 2.3e-10 | yes |
| lighthouse | d7 graded | gpu jaxamg | gpu jaxamg | yes | 2.7e-10 | yes |
| lighthouse | d7 nitsche | jaxamg ls 1 | jaxamg | yes | - | yes |
| lighthouse | d7 surface | gpu jaxamg | gpu jaxamg | yes | 3.4e-10 | yes |
| lighthouse | d8 dofs | gpu jaxamg | gpu jaxamg | yes | 2.6e-04 | NO |
| lighthouse | d8 graded | gpu jaxamg | gpu jaxamg | yes | 4.7e-10 | yes |
| lighthouse | d8 nitsche | jaxamg ls 1 | jaxamg | yes | - | yes |
| lighthouse | d8 surface | cpu (modified Newton) | cpu (modified Newton) | yes | 4.3e-10 | yes |
| loaf-of-bread | d7 dofs | gpu jaxamg | gpu jaxamg | yes | 2.4e-10 | yes |
| loaf-of-bread | d7 graded | gpu jaxamg | gpu jaxamg | yes | 5.3e-10 | yes |
| loaf-of-bread | d7 nitsche | jaxamg ls 5 | jaxamg | yes | - | yes |
| loaf-of-bread | d7 surface | gpu jaxamg | gpu jaxamg | yes | 4.0e-10 | yes |
| loaf-of-bread | d8 dofs | gpu jaxamg | gpu amgx | yes | 1.8e-09 | yes |
| loaf-of-bread | d8 graded | gpu jaxamg | gpu amgx | yes | 9.6e-10 | yes |
| loaf-of-bread | d8 nitsche | direct_continuation_session_tol ls 10 rtol 0.01 | pardiso-direct ls 10 rtol 0.01 | yes | - | yes |
| loaf-of-bread | d8 surface | lean-gpu | lean-gpu | yes | 8.7e-10 | yes |
| log-cabin | d7 dofs | gpu jaxamg | gpu jaxamg | yes | 9.3e-10 | yes |
| log-cabin | d7 graded | gpu jaxamg | gpu jaxamg | yes | 6.4e-10 | yes |
| log-cabin | d7 nitsche | jaxamg ls 1 | jaxamg | yes | - | yes |
| log-cabin | d7 surface | gpu jaxamg | gpu jaxamg | yes | 2.9e-10 | yes |
| log-cabin | d8 dofs | gpu jaxamg | gpu jaxamg | yes | 3.4e-10 | yes |
| log-cabin | d8 graded | gpu jaxamg | gpu jaxamg | yes | 8.1e-10 | yes |
| log-cabin | d8 nitsche | jaxamg ls 1 | jaxamg | yes | - | yes |
| log-cabin | d8 surface | cpu (modified Newton) | cpu (modified Newton) | yes | 3.4e-12 | yes |
| lumbar-vertebra | d7 dofs | gpu jaxamg | gpu jaxamg | yes | 4.8e-08 | yes |
| lumbar-vertebra | d7 graded | gpu jaxamg | gpu jaxamg | yes | 1.1e-08 | yes |
| lumbar-vertebra | d7 nitsche | jaxamg ls 1 | jaxamg ls 1 | yes | - | yes |
| lumbar-vertebra | d7 surface | gpu jaxamg | gpu jaxamg | yes | 2.6e-08 | yes |
| lumbar-vertebra | d8 dofs | gpu jaxamg | gpu jaxamg | yes | 1.8e-08 | yes |
| lumbar-vertebra | d8 graded | gpu jaxamg | gpu jaxamg | yes | 4.5e-09 | yes |
| lumbar-vertebra | d8 nitsche | jaxamg ls 1 | jaxamg ls 1 | yes | - | yes |
| lumbar-vertebra | d8 surface | gpu jaxamg | gpu jaxamg | yes | 1.1e-08 | yes |
| natural-stone-arch | d7 dofs | gpu jaxamg | gpu jaxamg | yes | 9.8e-10 | yes |
| natural-stone-arch | d7 graded | gpu jaxamg | gpu jaxamg | yes | 3.4e-10 | yes |
| natural-stone-arch | d7 nitsche | jaxamg ls 1 | jaxamg | yes | - | yes |
| natural-stone-arch | d7 surface | gpu jaxamg | gpu jaxamg | yes | 2.2e-10 | yes |
| natural-stone-arch | d8 dofs | gpu jaxamg | gpu jaxamg | yes | 6.1e-09 | yes |
| natural-stone-arch | d8 graded | gpu jaxamg | gpu jaxamg | yes | 6.2e-09 | yes |
| natural-stone-arch | d8 nitsche | jaxamg ls 1 | jaxamg | yes | - | yes |
| natural-stone-arch | d8 surface | cpu (modified Newton) | cpu (modified Newton) | yes | 9.0e-03 | NO |
| pantheon | d7 dofs | gpu jaxamg | gpu jaxamg | yes | 8.6e-12 | yes |
| pantheon | d7 graded | gpu jaxamg | gpu jaxamg | yes | 5.2e-11 | yes |
| pantheon | d7 nitsche | jaxamg ls 1 | jaxamg | yes | - | yes |
| pantheon | d7 surface | gpu jaxamg | gpu jaxamg | yes | 1.5e-10 | yes |
| pantheon | d8 dofs | gpu jaxamg | gpu jaxamg | yes | 8.3e-11 | yes |
| pantheon | d8 graded | gpu jaxamg | gpu jaxamg | yes | 3.9e-11 | yes |
| pantheon | d8 nitsche | jaxamg ls 1 | jaxamg | yes | - | yes |
| pantheon | d8 surface | cpu (modified Newton) | cpu (modified Newton) | yes | 1.7e-12 | yes |
| park-bench | d7 dofs | failed: generated surface self-intersects (TetGen -d); not repaired | failed: generated surface self-intersects (TetGen -d); not repaired | yes | - | yes |
| park-bench | d7 graded | failed: generated surface self-intersects (TetGen -d); not repaired | failed: generated surface self-intersects (TetGen -d); not repaired | yes | - | yes |
| park-bench | d7 nitsche | jaxamg ls 1 | jaxamg | yes | - | yes |
| park-bench | d7 surface | failed: generated surface self-intersects (TetGen -d); not repaired | failed: generated surface self-intersects (TetGen -d); not repaired | yes | - | yes |
| park-bench | d8 dofs | failed: generated surface self-intersects (TetGen -d); not repaired | failed: generated surface self-intersects (TetGen -d); not repaired | yes | - | yes |
| park-bench | d8 graded | failed: generated surface self-intersects (TetGen -d); not repaired | failed: generated surface self-intersects (TetGen -d); not repaired | yes | - | yes |
| park-bench | d8 nitsche | pyamgx ls 1 | pyamgx ls 1 | yes | - | yes |
| park-bench | d8 surface | failed: generated surface self-intersects (TetGen -d); not repaired | failed: generated surface self-intersects (TetGen -d); not repaired | yes | - | yes |
| pogo-stick | d7 dofs | gpu jaxamg | gpu jaxamg | yes | 6.5e-07 | yes |
| pogo-stick | d7 graded | gpu jaxamg | gpu jaxamg | yes | 5.2e-06 | yes |
| pogo-stick | d7 nitsche | jaxamg ls 1 | jaxamg ls 1 | yes | - | yes |
| pogo-stick | d7 surface | gpu jaxamg | gpu jaxamg | yes | 8.2e-07 | yes |
| pogo-stick | d8 dofs | gpu jaxamg_4000 | gpu jaxamg | yes | 7.1e-06 | yes |
| pogo-stick | d8 graded | gpu jaxamg_4000 | gpu jaxamg | yes | 4.5e-06 | yes |
| pogo-stick | d8 nitsche | jaxamg_4000 ls 1 | pyamgx ls 1 | yes | - | yes |
| pogo-stick | d8 surface | gpu jaxamg | gpu jaxamg | yes | 1.3e-05 | NO |
| rhinoceros | d7 dofs | gpu jaxamg | gpu jaxamg | yes | 2.7e-09 | yes |
| rhinoceros | d7 graded | gpu jaxamg | gpu jaxamg | yes | 2.3e-09 | yes |
| rhinoceros | d7 nitsche | jaxamg ls 1 | jaxamg ls 1 | yes | - | yes |
| rhinoceros | d7 surface | gpu jaxamg | gpu jaxamg | yes | 2.4e-09 | yes |
| rhinoceros | d8 dofs | gpu jaxamg | gpu jaxamg | yes | 5.7e-09 | yes |
| rhinoceros | d8 graded | gpu jaxamg | gpu jaxamg | yes | 8.6e-09 | yes |
| rhinoceros | d8 nitsche | jaxamg ls 1 | jaxamg ls 1 | yes | - | yes |
| rhinoceros | d8 surface | gpu jaxamg | gpu jaxamg | yes | 7.9e-09 | yes |
| road-roller-with-a-steel-drum-and-rubber-tyres | d7 dofs | gpu jaxamg | gpu jaxamg | yes | 1.0e-09 | yes |
| road-roller-with-a-steel-drum-and-rubber-tyres | d7 graded | gpu jaxamg | gpu jaxamg | yes | 1.0e-09 | yes |
| road-roller-with-a-steel-drum-and-rubber-tyres | d7 nitsche | jaxamg ls 1 | jaxamg ls 1 | yes | - | yes |
| road-roller-with-a-steel-drum-and-rubber-tyres | d7 surface | gpu jaxamg | gpu jaxamg | yes | 1.3e-09 | yes |
| road-roller-with-a-steel-drum-and-rubber-tyres | d8 dofs | gpu pyamgx | gpu jaxamg | yes | 4.3e-09 | yes |
| road-roller-with-a-steel-drum-and-rubber-tyres | d8 graded | gpu pyamgx | gpu jaxamg | yes | 8.7e-09 | yes |
| road-roller-with-a-steel-drum-and-rubber-tyres | d8 nitsche | jaxamg ls 1 | jaxamg ls 1 | yes | - | yes |
| road-roller-with-a-steel-drum-and-rubber-tyres | d8 surface | gpu pyamgx | gpu jaxamg | yes | 1.1e-08 | yes |
| school-chair-with-a-steel-frame-and-a-plastic-se | d7 dofs | gpu pyamgx | gpu jaxamg | yes | 6.1e-07 | yes |
| school-chair-with-a-steel-frame-and-a-plastic-se | d7 graded | gpu pyamgx | gpu jaxamg | yes | 7.0e-07 | yes |
| school-chair-with-a-steel-frame-and-a-plastic-se | d7 nitsche | jaxamg ls 1 | jaxamg ls 1 | yes | - | yes |
| school-chair-with-a-steel-frame-and-a-plastic-se | d7 surface | gpu pyamgx | gpu jaxamg | yes | 9.0e-08 | yes |
| school-chair-with-a-steel-frame-and-a-plastic-se | d8 dofs | gpu pyamgx | gpu amgx | yes | 3.5e-07 | yes |
| school-chair-with-a-steel-frame-and-a-plastic-se | d8 graded | gpu pyamgx | gpu amgx | yes | 9.1e-07 | yes |
| school-chair-with-a-steel-frame-and-a-plastic-se | d8 nitsche | pyamgx ls 1 | pyamgx ls 1 | yes | - | yes |
| school-chair-with-a-steel-frame-and-a-plastic-se | d8 surface | gpu pyamgx | gpu amgx | yes | 1.7e-05 | NO |
| skyscraper | d7 dofs | gpu jaxamg | gpu jaxamg | yes | 4.3e-10 | yes |
| skyscraper | d7 graded | gpu jaxamg | gpu jaxamg | yes | 2.9e-10 | yes |
| skyscraper | d7 nitsche | jaxamg ls 1 | jaxamg | yes | - | yes |
| skyscraper | d7 surface | gpu jaxamg | gpu jaxamg | yes | 7.5e-11 | yes |
| skyscraper | d8 dofs | gpu jaxamg | gpu jaxamg | yes | 2.1e-09 | yes |
| skyscraper | d8 graded | gpu jaxamg | gpu jaxamg | yes | 2.1e-09 | yes |
| skyscraper | d8 nitsche | jaxamg ls 1 | jaxamg | yes | - | yes |
| skyscraper | d8 surface | gpu jaxamg | gpu jaxamg | yes | 2.3e-09 | yes |
| slice-of-watermelon | d7 dofs | gpu jaxamg | gpu jaxamg | yes | 2.0e-08 | yes |
| slice-of-watermelon | d7 graded | gpu jaxamg | gpu jaxamg | yes | 8.0e-10 | yes |
| slice-of-watermelon | d7 nitsche | jaxamg ls 1 | jaxamg ls 1 | yes | - | yes |
| slice-of-watermelon | d7 surface | gpu jaxamg | gpu jaxamg | yes | 1.3e-09 | yes |
| slice-of-watermelon | d8 dofs | gpu jaxamg | gpu jaxamg | yes | 1.2e-09 | yes |
| slice-of-watermelon | d8 graded | gpu jaxamg | gpu jaxamg | yes | 1.8e-09 | yes |
| slice-of-watermelon | d8 nitsche | jaxamg ls 1 | jaxamg ls 1 | yes | - | yes |
| slice-of-watermelon | d8 surface | cpu (modified Newton) | cpu (modified Newton) | yes | 4.3e-10 | yes |
| sofa | d7 dofs | gpu jaxamg | gpu jaxamg | yes | 9.7e-10 | yes |
| sofa | d7 graded | gpu jaxamg | gpu jaxamg | yes | 9.2e-10 | yes |
| sofa | d7 nitsche | jaxamg ls 1 | jaxamg | yes | - | yes |
| sofa | d7 surface | gpu jaxamg | gpu jaxamg | yes | 7.4e-10 | yes |
| sofa | d8 dofs | gpu jaxamg | gpu jaxamg | yes | 2.2e-10 | yes |
| sofa | d8 graded | gpu jaxamg | gpu jaxamg | yes | 1.2e-09 | yes |
| sofa | d8 nitsche | jaxamg ls 1 | jaxamg | yes | - | yes |
| sofa | d8 surface | cpu (modified Newton) | cpu (modified Newton) | yes | 1.0e-11 | yes |
| taj-mahal | d7 dofs | gpu jaxamg | gpu jaxamg | yes | 1.0e-10 | yes |
| taj-mahal | d7 graded | gpu jaxamg | gpu jaxamg | yes | 2.7e-11 | yes |
| taj-mahal | d7 nitsche | jaxamg ls 1 | jaxamg | yes | - | yes |
| taj-mahal | d7 surface | gpu jaxamg | gpu jaxamg | yes | 1.3e-10 | yes |
| taj-mahal | d8 dofs | gpu jaxamg | gpu jaxamg | yes | 1.1e-11 | yes |
| taj-mahal | d8 graded | gpu jaxamg | gpu jaxamg | yes | 8.4e-12 | yes |
| taj-mahal | d8 nitsche | jaxamg ls 1 | jaxamg | yes | - | yes |
| taj-mahal | d8 surface | cpu (modified Newton) | cpu (modified Newton) | yes | 2.9e-12 | yes |
| teddy-bear | d7 dofs | gpu jaxamg | gpu jaxamg | yes | 1.9e-09 | yes |
| teddy-bear | d7 graded | gpu jaxamg | gpu jaxamg | yes | 1.3e-09 | yes |
| teddy-bear | d7 nitsche | jaxamg ls 1 | jaxamg ls 1 | yes | - | yes |
| teddy-bear | d7 surface | gpu jaxamg | gpu jaxamg | yes | 3.2e-09 | yes |
| teddy-bear | d8 dofs | gpu jaxamg | gpu jaxamg | yes | 8.4e-10 | yes |
| teddy-bear | d8 graded | gpu jaxamg | gpu jaxamg | yes | 3.8e-10 | yes |
| teddy-bear | d8 nitsche | jaxamg ls 1 | jaxamg ls 1 | yes | - | yes |
| teddy-bear | d8 surface | cpu (full Newton) | cpu (full Newton) | yes | 2.2e-12 | yes |
| thatched-stone-cottage | d7 dofs | gpu jaxamg | gpu jaxamg | yes | 1.0e-09 | yes |
| thatched-stone-cottage | d7 graded | gpu jaxamg | gpu jaxamg | yes | 1.8e-10 | yes |
| thatched-stone-cottage | d7 nitsche | jaxamg ls 1 | jaxamg | yes | - | yes |
| thatched-stone-cottage | d7 surface | gpu jaxamg | gpu jaxamg | yes | 4.6e-10 | yes |
| thatched-stone-cottage | d8 dofs | gpu jaxamg | gpu jaxamg | yes | 4.3e-11 | yes |
| thatched-stone-cottage | d8 graded | gpu jaxamg | gpu jaxamg | yes | 1.3e-10 | yes |
| thatched-stone-cottage | d8 nitsche | jaxamg ls 1 | jaxamg | yes | - | yes |
| thatched-stone-cottage | d8 surface | cpu (modified Newton) | cpu (modified Newton) | yes | 5.7e-12 | yes |
| truck-leaf-spring | d7 dofs | gpu jaxamg | gpu jaxamg | yes | 5.5e-09 | yes |
| truck-leaf-spring | d7 graded | gpu jaxamg | gpu jaxamg | yes | 7.5e-09 | yes |
| truck-leaf-spring | d7 nitsche | jaxamg ls 1 | jaxamg ls 1 | yes | - | yes |
| truck-leaf-spring | d7 surface | gpu jaxamg | gpu jaxamg | yes | 7.9e-09 | yes |
| truck-leaf-spring | d8 dofs | gpu jaxamg | gpu jaxamg | yes | 7.1e-09 | yes |
| truck-leaf-spring | d8 graded | gpu jaxamg | gpu jaxamg | yes | 2.2e-09 | yes |
| truck-leaf-spring | d8 nitsche | jaxamg ls 1 | jaxamg ls 1 | yes | - | yes |
| truck-leaf-spring | d8 surface | gpu jaxamg | gpu jaxamg | yes | 3.6e-09 | yes |
| tv-remote-control | d7 dofs | gpu jaxamg | gpu jaxamg | yes | 7.5e-10 | yes |
| tv-remote-control | d7 graded | gpu jaxamg | gpu jaxamg | yes | 6.3e-10 | yes |
| tv-remote-control | d7 nitsche | jaxamg ls 1 | jaxamg ls 1 | yes | - | yes |
| tv-remote-control | d7 surface | gpu jaxamg | gpu jaxamg | yes | 5.4e-10 | yes |
| tv-remote-control | d8 dofs | gpu jaxamg | gpu jaxamg | yes | 2.6e-09 | yes |
| tv-remote-control | d8 graded | gpu jaxamg | gpu jaxamg | yes | 2.5e-09 | yes |
| tv-remote-control | d8 nitsche | jaxamg ls 1 | jaxamg ls 1 | yes | - | yes |
| tv-remote-control | d8 surface | gpu jaxamg | gpu jaxamg | yes | 3.0e-09 | yes |
| usb-stick | d7 dofs | gpu jaxamg | gpu jaxamg | yes | 3.4e-08 | yes |
| usb-stick | d7 graded | gpu jaxamg | gpu jaxamg | yes | 3.5e-08 | yes |
| usb-stick | d7 nitsche | jaxamg ls 1 | jaxamg ls 1 | yes | - | yes |
| usb-stick | d7 surface | gpu jaxamg | gpu jaxamg | yes | 3.0e-08 | yes |
| usb-stick | d8 dofs | gpu jaxamg | gpu jaxamg | yes | 6.9e-10 | yes |
| usb-stick | d8 graded | gpu jaxamg | gpu jaxamg | yes | 1.0e-09 | yes |
| usb-stick | d8 nitsche | jaxamg ls 1 | jaxamg ls 1 | yes | - | yes |
| usb-stick | d8 surface | gpu jaxamg | gpu jaxamg | yes | 2.5e-10 | yes |
| walking-cane-with-a-rubber-tip | d7 dofs | gpu jaxamg | gpu jaxamg | yes | 5.3e-09 | yes |
| walking-cane-with-a-rubber-tip | d7 graded | gpu jaxamg | gpu jaxamg | yes | 5.4e-08 | yes |
| walking-cane-with-a-rubber-tip | d7 nitsche | jaxamg ls 1 | jaxamg | yes | - | yes |
| walking-cane-with-a-rubber-tip | d7 surface | gpu jaxamg | gpu jaxamg | yes | 3.7e-09 | yes |
| walking-cane-with-a-rubber-tip | d8 dofs | gpu jaxamg | gpu jaxamg | yes | 2.3e-07 | yes |
| walking-cane-with-a-rubber-tip | d8 graded | gpu jaxamg | gpu jaxamg | yes | 1.2e-07 | yes |
| walking-cane-with-a-rubber-tip | d8 nitsche | jaxamg ls 1 | jaxamg | yes | - | yes |
| walking-cane-with-a-rubber-tip | d8 surface | gpu jaxamg | gpu jaxamg | yes | 5.1e-08 | yes |
| wooden-rocking-horse | d7 dofs | gpu jaxamg | gpu jaxamg | yes | 2.1e-09 | yes |
| wooden-rocking-horse | d7 graded | gpu jaxamg | gpu jaxamg | yes | 4.4e-09 | yes |
| wooden-rocking-horse | d7 nitsche | jaxamg ls 1 | jaxamg | yes | - | yes |
| wooden-rocking-horse | d7 surface | gpu jaxamg | gpu jaxamg | yes | 3.3e-09 | yes |
| wooden-rocking-horse | d8 dofs | gpu jaxamg_4000 | gpu jaxamg | yes | 1.1e-08 | yes |
| wooden-rocking-horse | d8 graded | gpu jaxamg_4000 | gpu jaxamg | yes | 1.3e-08 | yes |
| wooden-rocking-horse | d8 nitsche | jaxamg ls 1 | jaxamg | yes | - | yes |
| wooden-rocking-horse | d8 surface | gpu jaxamg_4000 | gpu jaxamg | yes | 1.1e-08 | yes |
| wooden-toy-train-engine | d7 dofs | gpu jaxamg | gpu jaxamg | yes | 6.1e-10 | yes |
| wooden-toy-train-engine | d7 graded | gpu jaxamg | gpu jaxamg | yes | 4.2e-10 | yes |
| wooden-toy-train-engine | d7 nitsche | jaxamg ls 1 | jaxamg ls 1 | yes | - | yes |
| wooden-toy-train-engine | d7 surface | gpu jaxamg | gpu jaxamg | yes | 6.5e-10 | yes |
| wooden-toy-train-engine | d8 dofs | gpu jaxamg | gpu jaxamg | yes | 2.1e-10 | yes |
| wooden-toy-train-engine | d8 graded | gpu jaxamg | gpu jaxamg | yes | 1.4e-10 | yes |
| wooden-toy-train-engine | d8 nitsche | jaxamg ls 1 | jaxamg ls 1 | yes | - | yes |
| wooden-toy-train-engine | d8 surface | cpu (modified Newton) | cpu (modified Newton) | yes | 3.5e-11 | yes |
| yo-yo | d7 dofs | gpu jaxamg | gpu jaxamg | yes | 2.8e-10 | yes |
| yo-yo | d7 graded | gpu jaxamg | gpu jaxamg | yes | 3.4e-10 | yes |
| yo-yo | d7 nitsche | jaxamg ls 1 | jaxamg | yes | - | yes |
| yo-yo | d7 surface | gpu jaxamg | gpu jaxamg | yes | 5.2e-10 | yes |
| yo-yo | d8 dofs | cpu (modified Newton) | cpu (modified Newton) | yes | 1.4e-10 | yes |
| yo-yo | d8 graded | cpu (modified Newton) | cpu (modified Newton) | yes | 1.4e-10 | yes |
| yo-yo | d8 nitsche | jaxamg ls 1 | jaxamg | yes | - | yes |
| yo-yo | d8 surface | cpu (modified Newton) | cpu (modified Newton) | yes | 1.3e-10 | yes |
