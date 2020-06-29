epochs=50
round=10
p=1.0
train_size=1000

run_training() {
    python train.py --dataset cifar --alpha $1 --suffix ${2}_small --epochs $3 --p $4 --gpu $5 --train_size $train_size --no_save
}

> err_cifar.txt # Clear err.txt

for alpha in 3 1 0.3 0.1 0.01 0.003 0.001 0.0003 0.0001 0.00003 0.00001 0.0
do
    for i in 1 2 3 4 5 6 7 8 9 10
    do
        run_training $alpha $i $train_size $epochs $p $i 2>> err_cifar.txt &
    done
    wait
done
