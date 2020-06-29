epochs=100
round=5
p=1.0

run_mnist_training() {
    python mnist.py --alpha $1 --epochs $2 --gpu $3 --reg_type $4 --train_size $5 --model_type small --seed $6 --add_noise --alpha_noise 0.01
}
run_cifar_small_training() {
    python cifar10.py --alpha $1 --epochs $2 --gpu $3 --reg_type $4 --train_size $5 --model_type small --seed $6 --add_noise --alpha_noise 0.01

}
run_cifar_resnet_training() {
    python cifar10.py --alpha $1 --epochs $2 --gpu $3 --reg_type $4 --train_size $5 --model_type resnet --seed $6 --add_noise --alpha_noise 0.01

}

> err2.txt

for seed in 111 222 333 444 555 666 777 888 999 1111
do
for train_size in 50 100 200
do
for reg in db jac fob myfob1 myfob2
#for reg in db
do
    count=1
    for alpha in 3. 1. 0.3 0.1 0.03 0.01 0.003 0.001 0.0003 0.0001
    do
            run_mnist_training $alpha $epochs $count $reg $train_size $seed 2>> err2.txt &
            count=$[count+1]
    done
    count=1
    for alpha in 3. 1. 0.3 0.1 0.03 0.01 0.003 0.001 0.0003 0.0001
    do
            run_cifar_small_training $alpha $epochs $count $reg $train_size $seed 2>> err2.txt &
            count=$[count+1]
    done
    count=1
    for alpha in 3. 1. 0.3 0.1 0.03 0.01 0.003 0.001 0.0003 0.0001
    do
            run_cifar_resnet_training $alpha $epochs $count $reg $train_size $seed 2>> err2.txt &
            count=$[count+1]
    done
    wait
done
done
done
