network1 = {
    'Asia': makenode('Asia', [], '', 0.05),
    'Smoke': makenode('Smoke', [], '', 0.3),
    'TBC': makenode('TBC', ['Asia'], 't', 0.01, 'f', 0.001),
    'LC': makenode('LC', ['Smoke'], 't', 0.2, 'f', 0.08),
    'Bron': makenode('Bron', ['Smoke'], 't', 0.4, 'f', 0.1),
    'Xray': makenode('Xray', ['TBC', 'LC'], 'tt', 0.98, 'tf', 0.94, 'ft', 0.92, 'ff', 0.02),
    'Dysp': makenode('Dysp', ['TBC', 'LC', 'Bron'],
                     'ttt', 0.99, 'ttf', 0.97, 'tft', 0.98, 'tff', 0.9,
                     'ftt', 0.98, 'ftf', 0.92, 'fft', 0.95, 'fff', 0.07),
}