while getopts 'c:s:' OPTION; do
	case "$OPTION" in
		c)
			chr="$OPTARG"
			;;
		s)
			seed="$OPTARG"
	esac
done	


for workers in 2 5 10 50 100
	do
	echo "Workers running: $workers"
	mpiexec -n $workers python3 ./Jacob_Miller_HW2.py -c $chr -s $seed
done
