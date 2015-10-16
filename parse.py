import arrow

if __name__ == "__main__":
    with open("sched") as sched_file:
        for line in sched_file:
            print line.strip()
            print arrow.get(line.strip()[4:], "MMM D HH:mm:ss YYYY Z")

            #"Fri Oct 16 00:48:51 2015 -0000"
            #'Thu Oct 15 17:41:19 2015 -0700'
