import argparse
import os
import shlex
import subprocess
import time


def main(inp_dir, out_dir, start, end, k):
    # basedir = '/dfs/scratch2/prabhat8/trials-data/parsed_data_new/medex_input_json'
    basedir = inp_dir
    nctdirs = sorted(list(os.listdir(basedir)))
    print(len(nctdirs))
    end = min(end, len(nctdirs))
    classpath = '/afs/cs.stanford.edu/u/prabhat8/clinical-trials/medex/Medex_UIMA_1.3.8/bin:/afs/cs.stanford.edu/u/prabhat8/clinical-trials/medex/Medex_UIMA_1.3.8/lib/*'
    args_template = "java -Xmx1024m -cp {0} org.apache.medex.Main -i {1} -o {2} -b n -f y -d y -t n"
    # output_path_base = '/dfs/scratch2/prabhat8/trials-data/parsed_data_new/medex_output_json'
    output_path_base = out_dir
    curr = start
    pdict = {}
    processes = []
    while curr <= end:
        while len(processes) < k and curr <= end:
            nctdir = nctdirs[curr]
            output_path = os.path.join(output_path_base, nctdir[:-5])
            print(os.path.join(output_path_base, nctdir), os.path.exists(os.path.join(output_path_base, nctdir)))
            if os.path.exists(os.path.join(output_path_base, nctdir)):
                if curr % 10 == 0:
                    print(curr)
                curr += 1
                continue
            os.makedirs(output_path, exist_ok=True)
            args = args_template.format(classpath, os.path.join(basedir, nctdir), output_path)
            print(args)
            # args = args_template
            p = subprocess.Popen(shlex.split(args))
            pdict[p] = nctdir
            processes.append(p)
            time.sleep(10)
            curr += 1
        time.sleep(1 * 60)
        completed = []
        for p in processes:
            retcode = p.poll()
            if retcode is not None:
                p.wait()
                completed.append(p)
        if len(completed) > 0:
            print("=" * 40, "Completed following dirs", "=" * 40)
            for p in completed:
                processes.remove(p)
                print(pdict[p])
            print("=" * 120)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Medex Runner')
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--start', type=int, help='', required=True)
    parser.add_argument('--end', type=int, help='', required=True)
    parser.add_argument('--k', type=int, help='', required=True)

    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.start, args.end, args.k)
