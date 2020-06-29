epochs=100
round=5
p=1.0
train_size=1000

run_training() {
    python train.py --dataset fashion --alpha $1 --suffix $2 --epochs $3 --p $4 --gpu $5 --train_size $train_size --small
}

> err_fashion.txt # Clear err.txt

<<<<<<< HEAD
for alpha in 1. 0.3 0.1 0.03 0.01 0.003 0.001 0.0003 0.0001 0.00003 0.00001 0.
=======
for alpha in 1.0 0.1 0.03 0.01 0.003 0.001 0.0003 0.0001 0.00003 0.00001 0.
>>>>>>> d6b8822687b4265dc6414e7edc64cb4c6cf4579f
do
    for i in 6 7 8 9 10
    do
        run_training $alpha $i$train_size $epochs $p $i 2>> err_fashion.txt &
    done
    wait
done
