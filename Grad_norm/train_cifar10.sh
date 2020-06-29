epochs=50
round=5
p=1.0
#train_size=200

run_training() {
    python cifar10.py --alpha $1 --epochs $2 --gpu $3 --reg_type $4 --train_size $5 --model_type small
}

run_training_big() {
    python cifar10.py --alpha $1 --epochs $2 --gpu $3 --reg_type $4 --train_size $5 --model_type resnet
}

> err.txt

for train_size in 50 100 200
do
for reg in db jac fob myfob1 myfob2 NI
do
    count=1
    for alpha in 3. 1. 0.3 0.1 0.03 0.01 0.003 0.001 0.0003 0.0001
    do
            run_training $alpha $epochs $count $reg $train_size 2>> err.txt &
            count=$[count+1]
    done
    wait
done
done


for train_size in 50 100 200
do
for reg in db jac fob myfob1 myfob2 NI
do
    count=1
    for alpha in 3. 1. 0.3 0.1 0.03 0.01 0.003 0.001 0.0003 0.0001
    do
            run_training_big $alpha $epochs $count $reg $train_size 2>> err.txt &
            count=$[count+1]
    done
    wait
done
done
