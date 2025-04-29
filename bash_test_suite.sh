#!/bin/bash
# Usage: source bash_test_suite.sh filename
#        or
#        bash   bash_test_suite.sh filename
# where filename is for status output.

outfile=$1
rm -f $outfile

fftx_test() {
    # Set truncline to input line without initial white space.
    truncline="$(echo -e "${1}" | sed -e 's/^[[:space:]]*//')"
    if [[ -z $truncline ]] || [[ ${truncline::1} == "#" ]]; then
        # Copy blank line or comment.
        echo "$1"
    else
        start_time=`date +%s.%N`
        eval $1
        rc=$?
        end_time=`date +%s.%N`
        runtime=$( echo "$end_time - $start_time" | bc -l )
        if [[ $rc = 0 ]]; then
            echo "PASSED $1 in $runtime sec" >> $outfile
        else
            echo "FAILED $1 with code $rc in $runtime sec" >> $outfile
        fi
    fi
}

{
    all_start_time=`date +%s.%N`
    while read line; do
        fftx_test "$line"
    done < test_suite.sh
    all_end_time=`date +%s.%N`
    all_runtime=$( echo "$all_end_time - $all_start_time" | bc -l )
    echo "TOTAL TIME $all_runtime sec" >> $outfile
}
