#include <thread>
#include <iostream>
#include <queue>
#include <mutex>
#include <condition_variable> 

std::mutex mx;
std::condition_variable cv;
std::queue<int> q;

bool finished = false;

void producer(int n) {
	for(int i=0; i<n; ++i) {
		{
			std::lock_guard<std::mutex> lk(mx);//lock
			q.push(i);//push數值
			std::cout << "pushing " << i << std::endl;
			cv.notify_all();//會通知cv.wait 權限已釋出
		}//UNlock-建構子與結構子
		
	}
	{
		std::lock_guard<std::mutex> lk(mx);//lock
		finished = true;//設定的組數完成
	}//unlock
	cv.notify_all();//會通知cv.wait 權限已釋出
}

void consumer() {
	while (true) {
 
		std::unique_lock<std::mutex> lk(mx);//lock
		cv.wait(lk, []{ return finished || !q.empty(); });//與cv.notify_all()為一組，若數字還沒有PUSH到q內，不動作，啟動條件為q非空的或者數值都已PUSH完成
   
		while (!q.empty()) {
			std::cout << "consuming " << q.front() << std::endl;//front
			q.pop();//
		}
   
		if (finished) break;//數值都已PUSH完成-pop也完成-退出
	}
}

int main() {
	std::thread t1(producer, 50);//給予一個thread使用
	std::thread t2(consumer);//給予另一個thread使用
	t1.join();//確保結束
	t2.join();//確保結束
	std::cout << "finished!" << std::endl;
}
