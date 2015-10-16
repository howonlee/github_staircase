import arrow
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    seconds = []
    with open("/home/curuinor/data/linux_sched") as sched_file:
        sched_lines = sched_file.readlines()
        curr_date = arrow.get(sched_lines[0].strip()[4:], "MMM D HH:mm:ss YYYY Z")
        for line in sched_lines[1:]:
            next_date = arrow.get(line.strip()[4:], "MMM D HH:mm:ss YYYY Z")
            diff = curr_date - next_date
            if diff.days != 0:
                curr_date = next_date
                continue
            seconds.append(diff.seconds)
            curr_date = next_date
            #"Fri Oct 16 00:48:51 2015 -0000"
            #'Thu Oct 15 17:41:19 2015 -0700'
    plt.hist(seconds, log=True)
    plt.show()
