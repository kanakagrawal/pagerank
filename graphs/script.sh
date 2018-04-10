for f in *.csv; do
 	echo "File -> $f"
 	python3 profile.py $f
done