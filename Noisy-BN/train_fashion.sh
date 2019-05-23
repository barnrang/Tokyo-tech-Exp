epochs=100
round=5
p=1.0
train_size=1000

run_training() {
    python train.py --dataset fashion --alpha $1 --suffix $2 --epochs $3 --p $4 --gpu $5 --train_size $train_size
}

> err.txt # Clear err.txt

for alpha in 0.1 0.03 0.01 0.003 0.001 0.0003 0.0001 0.00003 0.00001 0.
do
    for i in 6 7 8 9 10
    do
        run_training $alpha $i$train_size $epochs $p $i 2>> err.txt &
    done
    wait
done
