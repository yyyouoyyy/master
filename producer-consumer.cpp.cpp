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
			q.push(i);//push�ƭ�
			std::cout << "pushing " << i << std::endl;
			cv.notify_all();//�|�q��cv.wait �v���w���X
		}//UNlock-�غc�l�P���c�l
		
	}
	{
		std::lock_guard<std::mutex> lk(mx);//lock
		finished = true;//�]�w���ռƧ���
	}//unlock
	cv.notify_all();//�|�q��cv.wait �v���w���X
}

void consumer() {
	while (true) {
 
		std::unique_lock<std::mutex> lk(mx);//lock
		cv.wait(lk, []{ return finished || !q.empty(); });//�Pcv.notify_all()���@�աA�Y�Ʀr�٨S��PUSH��q���A���ʧ@�A�Ұʱ���q�D�Ū��Ϊ̼ƭȳ��wPUSH����
   
		while (!q.empty()) {
			std::cout << "consuming " << q.front() << std::endl;//front
			q.pop();//
		}
   
		if (finished) break;//�ƭȳ��wPUSH����-pop�]����-�h�X
	}
}

int main() {
	std::thread t1(producer, 50);//�����@��thread�ϥ�
	std::thread t2(consumer);//�����t�@��thread�ϥ�
	t1.join();//�T�O����
	t2.join();//�T�O����
	std::cout << "finished!" << std::endl;
}
