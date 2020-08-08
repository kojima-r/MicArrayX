import subprocess
import sys
from optparse import OptionParser


def main():
    usage = "usage: %s tf [options] <in: src.wav> <out: dest.wav>" % sys.argv[0]
    parser = OptionParser()
    parser.add_option(
        "-t",
        "--tf",
        dest="tf",
        help="tf.zip(HARK2 transfer function file>",
        default=None,
        type=str,
        metavar="FILE",
    )

    parser.add_option(
        "-d",
        "--direction",
        dest="direction",
        help="arrival direction of sound (degree)",
        default=None,
        type=str,
    )

    parser.add_option(
        "-s",
        "--script",
        dest="script",
        help="temporary script",
        default="./const_sep.n",
        type=str,
    )

    parser.add_option(
        "-i",
        "--input",
        dest="org_script",
        help="script",
        default="./const_sep.n.tmpl",
        type=str,
    )

    (options, args) = parser.parse_args()

    # Separation
    f = args[0]

    i_file = open(options.org_script)
    o_file = open(options.script, "w")
    direction = list(map(float, options.direction.split(",")))
    print(f)
    # Replace
    lines = i_file.readlines()
    arr_ang = direction
    arr_ele = [0 for i in range(len(direction))]
    for line in lines:
        if line.find('<Parameter name="ANGLES"') >= 0:
            line = line.replace(
                "pmt1",
                "&lt;Vector&lt;float&gt; " + " ".join(map(str, arr_ang)) + " &gt;",
            )
            print(line)
        if line.find('<Parameter name="ELEVATIONS"') >= 0:
            line = line.replace(
                "pmt2",
                "&lt;Vector&lt;float&gt; " + " ".join(map(str, arr_ele)) + " &gt;",
            )
            print(line)
        if line.find('<Parameter name="TF_CONJ_FILENAME" ') >= 0:
            line = line.replace("pmt3", options.tf)
            print(line)
        o_file.write(line)
    o_file.close()
    i_file.close()

    subprocess.call(["batchflow", options.script, f])

if __name__ == "__main__":
    main()
