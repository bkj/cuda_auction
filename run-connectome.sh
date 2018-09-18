
datadir="/home/bjohnson/projects/sgm/_data/connectome"

make clean; make
for dataset in $(ls $datadir | head -n 15 | tail -n 1); do
    echo '------------------------------'
    echo $dataset
    path="$datadir/$dataset/sparse/graph"
    cp $path graph
    echo '---'
    echo 'auction'
    ./main > res
    # echo "---"
    # echo "reference"
    # python reference.py
done


for dataset in $(ls $datadir | head -n 15 | tail -n 1); do
    echo '------------------------------'
    echo $dataset
    path="$datadir/$dataset/sparse/graph"
    cp $path graph
    # echo '---'
    # echo 'auction'
    # ./main > res
    echo "---"
    echo "reference"
    python reference.py
done
