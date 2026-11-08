"""Exact-version readers for released event/person counterparts; unavailable tokens stay unavailable."""
import csv
import hashlib
from pathlib import Path
from countviews.values import parse_count

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def fold(code):
    """Fixed geographic partition, shared across periods and families."""
    return int(hashlib.sha256(code.encode()).hexdigest()[:8], 16) % 5

def read_pairs(config):
    pairs = []
    provenance = []
    for source in config['sources']:
        path = Path(source['path'])
        actual = digest(path)
        if actual != source['sha256']:
            raise ValueError(f'Input changed: {path}')
        provenance.append(dict(path=str(path), sha256=actual))
        with path.open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        if source['format'] == 'adult_requests':
            totals, clients = {}, {}
            for row in rows:
                if row['AgeBand'] not in ['18 to 64', '65 and over']:
                    continue
                code = row['GeographyCode'] or row['RegionGOCode']
                key = (row['FYEnding'], code, row['AgeBand'])
                if row['AggregationLevel'] == 'Age band total':
                    target = totals
                elif row['AggregationLevel'] == 'Client Count':
                    target = clients
                else:
                    continue
                if key in target:
                    raise ValueError(f'Duplicate adult key: {key}')
                target[key] = row
            if totals.keys() != clients.keys():
                raise ValueError('Adult event/client keys differ')
            for key, event in totals.items():
                person = clients[key]
                period, code, age = key
                pairs.append(dict(source='DHSC', family='requests age '+age,
                    period=period, code=code,
                    level='Local authority' if event['GeographyLevel']=='Local Authority' else event['GeographyLevel'],
                    events=event['ITEMVALUE'], persons=person['ITEMVALUE'], rounding_tolerance=4,
                    path=str(path)))
        else:
            for row in rows:
                code = row['new_la_code'] or row['region_code'] or row['country_code']
                for family, event, person in source['pairs']:
                    pairs.append(dict(source='DfE', family=family,
                        period=row['time_period'], code=code, level=row['geographic_level'],
                        events=row[event], persons=row[person], rounding_tolerance=0 if row['geographic_level']=='Local authority' else 10,
                        path=str(path)))
    seen = set()
    for row in pairs:
        key = tuple(row[k] for k in ['source', 'family', 'period', 'level', 'code'])
        if key in seen:
            raise ValueError(f'Duplicate paired key {key}')
        seen.add(key)
    return pairs, provenance
