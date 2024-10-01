import os


def create_nctids_file(nctids, output_path, batch_size=1000):
    list_nctid = nctids.tolist()
    start = 0
    while start < len(list_nctid):
        i = start
        end = min(i + batch_size, len(list_nctid))
        with open(os.path.join(output_path, f'criteria_to_parse_{i}_{end}.txt'), 'w') as f:
            for nct_id in list_nctid[i:end]:
                f.write(f'{nct_id}\n')
        start += batch_size
