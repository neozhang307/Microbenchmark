for i in {1..10}
do
  echo "Running iteration $i..."
  ./globalltc 2> log  >> globalltc.log
done