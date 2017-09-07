#include "blur_video.h"
#include "encoder.h"
#include "muxer.h"
#include "yolo_detector.h"

#include <condition_variable>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <cstdio>

#include <glog/logging.h>
#include <opencv2/core/ocl.hpp>
#include <opencv2/opencv.hpp>
#include <sys/time.h>

#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cstring>
#include <ctime>

#define MAX_NUM_DECODING_FRAMES 32
#define MAX_NUM_ENCODING_FRAMES 32

using namespace cv;
using namespace std;

struct ImageAndBoxAndProb
{
	Mat image;
	box *boxes;
	float **probabilities;
};

volatile static int g_curr_frame_idx = INT_MAX;
volatile static bool g_decoding_is_finished = false;

static mutex g_dq_mtx;
static condition_variable g_dq_empty_cv;
static condition_variable g_dq_full_cv;

static mutex g_eq_mtx;
static condition_variable g_eq_empty_cv;
static condition_variable g_eq_full_cv;

static queue<ImageAndBoxAndProb> g_dec_q;
static queue<ImageAndBoxAndProb> g_enc_q;

struct track_state
{
	int det_count;
	int fail_count;
	int num; 
	int state;       
	Rect2d size_draw;
};



void tracking_function(Mat frame,Size vid_size, MultiTracker &Multi_trackers, vector<track_state>&draw_track ,vector<Rect2d> &Multi_objects,vector<YoloDetector::BoundingBox> bbox_list)
{


#if 0
        struct timespec tt1, tt2;
        double etime;

        clock_gettime(CLOCK_REALTIME, &tt1);

        clock_gettime(CLOCK_REALTIME, &tt2);
        LOG(INFO) << "got " << bbox_list_idx << " bboxes\n";

        etime = (double)(tt2.tv_sec - tt1.tv_sec) * 1000.0 + (double)(tt2.tv_nsec - tt1.tv_nsec) / 10000000.0;
        LOG(INFO) << "total time: " << etime << " ms, avg time: " << etime / (double)bbox_list_idx << " ms\n";
#endif




        struct timespec track_start,track_end,track_all_start,track_all_end;
        double etime;

        clock_gettime(CLOCK_REALTIME, &track_all_start);

	int YOLO_detect[50];
        int Multi_detect[50];




	memset(YOLO_detect,0,sizeof(YOLO_detect));
	memset(Multi_detect,0,sizeof(Multi_detect));


        clock_gettime(CLOCK_REALTIME, &track_start);

	//update the tracking result
        //流程圖內-Tracker update(藍框)
	if(Multi_objects.size()>0)
		Multi_trackers.update(frame,Multi_objects);


        clock_gettime(CLOCK_REALTIME, &track_end);
        etime = (double)(track_end.tv_sec - track_start.tv_sec) * 1000.0 + (double)(track_end.tv_nsec - track_start.tv_nsec) / 10000000.0;

        LOG(INFO) << "multi_num: " << Multi_objects.size() << " \n";

        LOG(INFO) << "track-1-time: " << etime << " ms \n";


        

        //tracker模式
	std::string trackingAlg = "KCF";    //BOOSTING>>KCF==MEDIANFLOW

	int YOLO_num=0;


        clock_gettime(CLOCK_REALTIME, &track_start);


	//////流程圖內-藍框內有紅框
	if(Multi_objects.size()>0)
	{
		for(int i=0;i<Multi_objects.size();i++)
		{ 

			YOLO_num=0;

			for (const auto &bbox: bbox_list) 
			{             


					if(((Multi_trackers.objects[i].x)<(bbox.x))
							&&((Multi_trackers.objects[i].x+Multi_trackers.objects[i].width)>(bbox.x+bbox.width))
							&&((Multi_trackers.objects[i].y)<(bbox.y))
							&&((Multi_trackers.objects[i].y+Multi_trackers.objects[i].height)>(bbox.y+bbox.height))
							&&(Multi_detect[i]==0)&&(YOLO_detect[YOLO_num]==0))
					{ 

						Multi_detect[i]=1;
						YOLO_detect[YOLO_num]=1;
						
                                                draw_track[i].det_count+=1; 
						draw_track[i].fail_count=0;
						draw_track[i].state=1;  

                                                ///////change-x,y,w,h/////////////////

						draw_track[i].size_draw.x=bbox.x-15;
						Multi_trackers.objects[i].x=bbox.x-15;
						draw_track[i].size_draw.y=bbox.y-15;
						Multi_trackers.objects[i].y=bbox.y-15;
						draw_track[i].size_draw.width=bbox.width+30;
						Multi_trackers.objects[i].width=bbox.width+30;
						draw_track[i].size_draw.height=bbox.height+30;
						Multi_trackers.objects[i].height=bbox.height+30;   

						Multi_objects[i].x=bbox.x-15;                    
						Multi_objects[i].y=bbox.y-15;                     
						Multi_objects[i].width=bbox.width+30;
						Multi_objects[i].height=bbox.height+30;                                            
					}


				YOLO_num++;
			}              
		}
	}



        clock_gettime(CLOCK_REALTIME, &track_end);
        etime = (double)(track_end.tv_sec - track_start.tv_sec) * 1000.0 + (double)(track_end.tv_nsec - track_start.tv_nsec) / 10000000.0;
        LOG(INFO) << "track-2-time: " << etime << " ms \n";



        clock_gettime(CLOCK_REALTIME, &track_start);



	////i程圖內-二次藍框內有紅框
	int tracker_state_change=0;

	for(int i=0;i<Multi_objects.size();i++)
	{

		if( Multi_detect[i]==0)
		{
			YOLO_num=0;

			int track_overlap=0;

			for (const auto &bbox: bbox_list) 
			{


					if(((draw_track[i].size_draw.x-draw_track[i].size_draw.width/2)<(bbox.x))
							&&((draw_track[i].size_draw.x+draw_track[i].size_draw.width*3/2)>(bbox.x+bbox.width))
							&&((draw_track[i].size_draw.y-draw_track[i].size_draw.height/2)<(bbox.y))
							&&((draw_track[i].size_draw.y+draw_track[i].size_draw.height*3/2)>(bbox.y+bbox.height))
							&&(YOLO_detect[YOLO_num]==0))


					{

						track_overlap=1;

						Multi_detect[i]=1;
						YOLO_detect[YOLO_num]=1;

						draw_track[i].det_count+=1; 
						draw_track[i].fail_count=0;     
						draw_track[i].state=1;  
			
                            			///////change-x,y,w,h/////////////////

						draw_track[i].size_draw.x=bbox.x-15;
						Multi_trackers.objects[i].x=bbox.x-15;
						draw_track[i].size_draw.y=bbox.y-15;
						Multi_trackers.objects[i].y=bbox.y-15;
						draw_track[i].size_draw.width=bbox.width+30;
						Multi_trackers.objects[i].width=bbox.width+30;
						draw_track[i].size_draw.height=bbox.height+30;
						Multi_trackers.objects[i].height=bbox.height+30;   

						Multi_objects[i].x=bbox.x-15;                    
						Multi_objects[i].y=bbox.y-15;                     
						Multi_objects[i].width=bbox.width+30;
						Multi_objects[i].height=bbox.height+30;                                                                                                                                   
					}

				YOLO_num++;
			}

                        
                        //如果在邊框,且沒有重疊到紅框則消除
			if((((draw_track[i].size_draw.x<5)||(draw_track[i].size_draw.x>(vid_size.width-5))||(draw_track[i].size_draw.y<5)||(draw_track[i].size_draw.y>(vid_size.height-5))
			||(draw_track[i].size_draw.x+draw_track[i].size_draw.width<5)||(draw_track[i].size_draw.x+draw_track[i].size_draw.width>(vid_size.width-5))
			||(draw_track[i].size_draw.y+draw_track[i].size_draw.height<5)||(draw_track[i].size_draw.y+draw_track[i].size_draw.height>(vid_size.height-5))
			||((draw_track[i].size_draw.x+draw_track[i].size_draw.width)<draw_track[i].size_draw.x)
			||((draw_track[i].size_draw.y+draw_track[i].size_draw.height)<draw_track[i].size_draw.y))&& (track_overlap==0)))
			{	
				tracker_state_change=1;
				draw_track[i].state=-1;//state=-1為準備消除
			}   
                        //沒有重疊到紅框計數,fail_count記數,若60次沒紅框資料,則消除
			if(track_overlap==0)
			{
				draw_track[i].fail_count+=1;
 
				if(draw_track[i].fail_count>60)
				{
					tracker_state_change=1;
					draw_track[i].state=-1;
				}
			}     

		}
	} 



        clock_gettime(CLOCK_REALTIME, &track_end);
        etime = (double)(track_end.tv_sec - track_start.tv_sec) * 1000.0 + (double)(track_end.tv_nsec - track_start.tv_nsec) / 10000000.0;
        LOG(INFO) << "track-3-time: " << etime << " ms \n";



        clock_gettime(CLOCK_REALTIME, &track_start);


	///clear-這邊會特別再做一次是因為MultiTracker好像沒辦法單一清除一個,之前清除失敗,所以才全部一次清除跟新增
	if(tracker_state_change==1)
	{	
#if 1
		Multi_trackers.objects.clear();         
		MultiTracker Multi_trackers2(trackingAlg);
		Multi_trackers=Multi_trackers2;

		for(int i=0;i<Multi_objects.size();i++)
		{

                        //清除tracker的部分 
			if(draw_track[i].state==-1)
			{
                                draw_track[i].state=0;
				Multi_objects.erase (Multi_objects.begin()+i);
				draw_track.erase (draw_track.begin()+i);       
				//Multi_objects.size()=Multi_objects.size()-1;
				i=i-1;
			}//保留tracker的部分
			else
			{
				Multi_trackers.add(frame,Multi_objects[i]);

			}


		}
#endif
#if 0

               // Multi_trackers.objects.clear();
               // MultiTracker Multi_trackers2(trackingAlg);
               // Multi_trackers=Multi_trackers2;

                for(int i=0;i<Multi_objects.size();i++)
                {

                        //清除tracker的部分
                        if(draw_track[i].state==-1)
                        {
                                draw_track[i].state=0;
                                Multi_objects.erase (Multi_objects.begin()+i);
                                draw_track.erase (draw_track.begin()+i);
                                Multi_trackers.erase(Multi_trackers.objects[i]);

                                //Multi_objects.size()=Multi_objects.size()-1;
                                i=i-1;
                        }//保留tracker的部分
                 //       else
                   //     {
    //                            Multi_trackers.add(frame,Multi_objects[i]);
//
  //                      }


                }
#endif

	}


        clock_gettime(CLOCK_REALTIME, &track_end);
        etime = (double)(track_end.tv_sec - track_start.tv_sec) * 1000.0 + (double)(track_end.tv_nsec - track_start.tv_nsec) / 10000000.0;
        LOG(INFO) << "track-4-time: " << etime << " ms \n";




        clock_gettime(CLOCK_REALTIME, &track_start);




	//////流程圖內-剩餘紅框新增藍框 (代表目前紅框沒有藍框資料) 
	YOLO_num=0;


	for (const auto &bbox: bbox_list) 
	{


		rectangle(frame, Point(bbox.x,bbox.y), Point(bbox.x+bbox.width,bbox.y+bbox.height), Scalar(0,0,255), 5);

                int multi_tmp=Multi_objects.size();

		if(YOLO_detect[YOLO_num]==0)
		{
			int add_x=bbox.x-15;
			int add_y=bbox.y-15;        
			int add_w=bbox.width+30;
			int add_h=bbox.height+30;   

                        Rect rect(add_x, add_y, add_w, add_h);


                        Multi_objects.push_back(rect);

			Multi_trackers.add(frame,Multi_objects[multi_tmp]);

			draw_track[multi_tmp].num=multi_tmp;  
			draw_track[multi_tmp].size_draw=Multi_objects[multi_tmp];
			draw_track[multi_tmp].det_count=0; 
			draw_track[multi_tmp].fail_count=0;     
			draw_track[multi_tmp].state=1;       



		       // Multi_objects.size()++;
			
		}

		YOLO_num++;

	}



        clock_gettime(CLOCK_REALTIME, &track_end);
        etime = (double)(track_end.tv_sec - track_start.tv_sec) * 1000.0 + (double)(track_end.tv_nsec - track_start.tv_nsec) / 10000000.0;
        LOG(INFO) << "track-5-time: " << etime << " ms \n";



        clock_gettime(CLOCK_REALTIME, &track_start);




	//把框畫出來
	for(int i=0;i<Multi_objects.size();i++)
	{

                //藍框內沒紅框
		if(draw_track[i].state==0)
		{
                        //畫出原始區塊
			/*
			   int xx=Multi_trackers.objects[i].x+15;
			   int yy=Multi_trackers.objects[i].y+15;

			   int ww=Multi_trackers.objects[i].width-30;
			   int hh=Multi_trackers.objects[i].height-30;   




			   rectangle(frame, Point(xx,yy), Point(xx+ww,yy+hh), Scalar(255,0,0), 5);           
			 */
			rectangle( frame, Multi_trackers.objects[i], Scalar( 255, 255, 0 ), 2, 1 );           

		}//藍框內有紅框
		else if(draw_track[i].state==1)
		{			   


			rectangle( frame, Multi_trackers.objects[i], Scalar( 255, 0, 0 ), 2, 1 );  
			 //畫出原始區塊
                         /*
			   int xx=Multi_trackers.objects[i].x+15;
			   int yy=Multi_trackers.objects[i].y+15;

			   int ww=Multi_trackers.objects[i].width-30;
			   int hh=Multi_trackers.objects[i].height-30;   



			   rectangle(frame, Point(xx,yy), Point(xx+ww,yy+hh), Scalar(255,0,0), 5);    
			 */         

		}

		draw_track[i].state=0;    


//                rectangle(frame, Multi_test[i], Scalar( 0, 255, 0 ), 2, 1 );

	}


        clock_gettime(CLOCK_REALTIME, &track_end);

        etime = (double)(track_end.tv_sec - track_start.tv_sec) * 1000.0 + (double)(track_end.tv_nsec - track_start.tv_nsec) / 10000000.0;
        LOG(INFO) << "track-6-time: " << etime << " ms \n";


        clock_gettime(CLOCK_REALTIME, &track_all_end);
        etime = (double)(track_all_end.tv_sec - track_all_start.tv_sec) * 1000.0 + (double)(track_all_end.tv_nsec - track_all_start.tv_nsec) / 10000000.0;
        LOG(INFO) << "track-all-time: " << etime << " ms\n";

}





static void encode_video_track(void *ptr, const char *invid, const char *outvid, int bitrate)
{
	YoloDetector *yolo_det = (YoloDetector*) ptr;
	int frame_idx = 0;

//--------------------------影片輸入
	VideoCapture cap(invid);
	Size vid_size = Size((int) cap.get(CV_CAP_PROP_FRAME_WIDTH), (int) cap.get(CV_CAP_PROP_FRAME_HEIGHT));
	double vid_fps = cap.get(CV_CAP_PROP_FPS);

	Muxer::Parameters mux_params = Muxer::GetDefaultParameters();
	Muxer muxer((int)round(vid_fps), 1, outvid, mux_params);

	Encoder::Parameters enc_params = Encoder::GetDefaultParameters();
	enc_params.global_header = muxer.IsGlobalHeaderRequired();
	enc_params.bframes = 4;
	enc_params.preset = "veryfast";
	enc_params.bitrate = bitrate * 1000;
	Encoder encoder(vid_size.width, vid_size.height, (int)round(vid_fps), 1, enc_params);

	cv::ocl::setUseOpenCL(false);

	AVRational enc_time_base = encoder.GetEncoderTimeBase();
	ImageAndBoxAndProb img_box;
	AVPacket packet = {0};

	muxer.CopyEncoderParameters(encoder.GetEncodeContext());
	muxer.WriteHeader();



//--------------------------一些tracking宣告
	// Set up tracker. 
	// Instead of MIL, you can also use 
	// BOOSTING, KCF, TLD, MEDIANFLOW or GOTURN  
	// Ptr<Tracker> tracker = Tracker::create( "MIL" );
	// set the default tracking algorithm
	std::string trackingAlg = "MEDIANFLOW";    //BOOSTING>>KCF==MEDIANFLOW
	// create the tracker
	MultiTracker Multi_trackers(trackingAlg);
	// container of the tracked objects
	vector<Rect2d> Multi_objects;	


	vector<track_state> draw_track;
	draw_track.resize(50);



//--------------------------開始讀影片 (frame)
	while (true) {
		{
			unique_lock<mutex> lck(g_eq_mtx);
			while (g_enc_q.empty() && g_curr_frame_idx > frame_idx)
				g_eq_empty_cv.wait(lck);

			if (g_curr_frame_idx <= frame_idx)
				break;

			img_box = g_enc_q.front();
			g_enc_q.pop();
			g_eq_full_cv.notify_one();
		}

		Mat frame = img_box.image;//frame為tracking處理的圖片


                //流程圖內-Yolo 偵測-(紅框)
		vector<YoloDetector::BoundingBox> bbox_list = yolo_det->PerformNMS(img_box.boxes, img_box.probabilities, frame.cols, frame.rows);
		yolo_det->FreeBoxAndProbability(img_box.boxes, img_box.probabilities);

//--------------------------tracking程式
		tracking_function(frame, vid_size, Multi_trackers, draw_track, Multi_objects, bbox_list);


		Mat frame_yuv420p;
		cvtColor(frame, frame_yuv420p, CV_BGR2YUV_I420);

		encoder.SendFrame(frame_yuv420p.data, frame_yuv420p.cols);
		av_init_packet(&packet);
		int ret = encoder.ReceivePacket(&packet);

		if (0 < ret)
			muxer.WritePacket(&packet, enc_time_base);

		++frame_idx;
		if (g_curr_frame_idx <= frame_idx)
			break;
	}

	encoder.SendFrame(NULL, 0);
	while (true) {
		av_init_packet(&packet);
		int ret = encoder.ReceivePacket(&packet);

		if (0 < ret)
			muxer.WritePacket(&packet, enc_time_base);
		else
			break;
	}

	muxer.WriteTrailer();
	LOG(INFO) << "encoded " << frame_idx << " frames\n";
}


//原始程式
#if 0
static void encode_video(void *ptr, const char *invid, const char *outvid, int bitrate)
{
	YoloDetector *yolo_det = (YoloDetector*) ptr;
	int frame_idx = 0;

	VideoCapture cap(invid);
	Size vid_size = Size((int) cap.get(CV_CAP_PROP_FRAME_WIDTH), (int) cap.get(CV_CAP_PROP_FRAME_HEIGHT));
	double vid_fps = cap.get(CV_CAP_PROP_FPS);

	Muxer::Parameters mux_params = Muxer::GetDefaultParameters();
	Muxer muxer((int)round(vid_fps), 1, outvid, mux_params);

	Encoder::Parameters enc_params = Encoder::GetDefaultParameters();
	enc_params.global_header = muxer.IsGlobalHeaderRequired();
	enc_params.bframes = 4;
	enc_params.preset = "veryfast";
	enc_params.bitrate = bitrate * 1000;
	Encoder encoder(vid_size.width, vid_size.height, (int)round(vid_fps), 1, enc_params);

	cv::ocl::setUseOpenCL(false);

	AVRational enc_time_base = encoder.GetEncoderTimeBase();
	ImageAndBoxAndProb img_box;
	AVPacket packet = {0};

	muxer.CopyEncoderParameters(encoder.GetEncodeContext());
	muxer.WriteHeader();
	while (true) {
		{
			unique_lock<mutex> lck(g_eq_mtx);
			while (g_enc_q.empty() && g_curr_frame_idx > frame_idx)
				g_eq_empty_cv.wait(lck);

			if (g_curr_frame_idx <= frame_idx)
				break;

			img_box = g_enc_q.front();
			g_enc_q.pop();
			g_eq_full_cv.notify_one();
		}

		Mat frame = img_box.image;
		vector<YoloDetector::BoundingBox> bbox_list = yolo_det->PerformNMS(img_box.boxes, img_box.probabilities, frame.cols, frame.rows);
		yolo_det->FreeBoxAndProbability(img_box.boxes, img_box.probabilities);

		for (const auto &bbox: bbox_list) {
			Rect region(bbox.x, bbox.y, bbox.width, bbox.height);
			GaussianBlur(frame(region), frame(region), Size(0, 0), 4);
		}

		Mat frame_yuv420p;
		cvtColor(frame, frame_yuv420p, CV_BGR2YUV_I420);

		encoder.SendFrame(frame_yuv420p.data, frame_yuv420p.cols);
		av_init_packet(&packet);
		int ret = encoder.ReceivePacket(&packet);

		if (0 < ret)
			muxer.WritePacket(&packet, enc_time_base);

		++frame_idx;
		if (g_curr_frame_idx <= frame_idx)
			break;
	}

	encoder.SendFrame(NULL, 0);
	while (true) {
		av_init_packet(&packet);
		int ret = encoder.ReceivePacket(&packet);

		if (0 < ret)
			muxer.WritePacket(&packet, enc_time_base);
		else
			break;
	}

	muxer.WriteTrailer();
	LOG(INFO) << "encoded " << frame_idx << " frames\n";
}
#endif

static void decode_video_and_send_frame(void *ptr, const char *vidfile)
{
	YoloDetector *yolo_det = (YoloDetector*) ptr;
	cv::VideoCapture cap(vidfile);
	int frame_idx = 0;

	while (true) {
		Mat frame;
		cap >> frame;
		if (!frame.data)
			break;

		yolo_det->SendImage(frame);
		++frame_idx;

		ImageAndBoxAndProb img_box;
		img_box.image = frame.clone();

		{
			unique_lock<mutex> lck(g_dq_mtx);
			while (MAX_NUM_DECODING_FRAMES == g_dec_q.size())
				g_dq_full_cv.wait(lck);

			g_dec_q.push(img_box);
			g_dq_empty_cv.notify_one();
		}
	}

	LOG(INFO) << "sent " << frame_idx << " frames\n";
	g_curr_frame_idx = frame_idx;

	{
		unique_lock<mutex> lck(g_eq_mtx);
		g_eq_empty_cv.notify_one();
	}

	{
		unique_lock<mutex> lck(g_dq_mtx);
		g_decoding_is_finished = true;
		g_dq_empty_cv.notify_one();
	}

}

void benchmark_blurring_video_threaded_v2(YoloDetector &yolo_det, const char *invid, const char *outvid, int bitrate)
{
	//g_curr_frame_idx = INT_MAX;
	struct timespec tt1, tt2;
	int bbox_list_idx = 0;
	double etime;

	clock_gettime(CLOCK_REALTIME, &tt1);
	thread dec_send_thread = thread(decode_video_and_send_frame, &yolo_det, invid);
	thread enc_thread = thread(encode_video_track, &yolo_det, invid, outvid, bitrate);

	while (true) {
		ImageAndBoxAndProb img_box;


		{
			// pop from decoding queue
			unique_lock<mutex> lck(g_dq_mtx);
			while (g_dec_q.empty() && !g_decoding_is_finished)
				g_dq_empty_cv.wait(lck);

			if (g_dec_q.empty() && g_decoding_is_finished)
				break;

			img_box = g_dec_q.front();
			g_dec_q.pop();
			g_dq_full_cv.notify_one();
		}

		float **probs = NULL;
		box *boxes = NULL;

		yolo_det.AllocateBoxAndProbability(&boxes, &probs);
		yolo_det.GetBoxAndProb(boxes, probs);

		img_box.probabilities = probs;
		img_box.boxes = boxes;

		{
			// push to encoding queue
			unique_lock<mutex> lck(g_eq_mtx);
			while (MAX_NUM_ENCODING_FRAMES == g_enc_q.size())
				g_eq_full_cv.wait(lck);

			g_enc_q.push(img_box);
			g_eq_empty_cv.notify_one();
		}

		++bbox_list_idx;
		if (g_curr_frame_idx <= bbox_list_idx)
			break;
	}

	dec_send_thread.join();
	enc_thread.join();
	clock_gettime(CLOCK_REALTIME, &tt2);
	LOG(INFO) << "got " << bbox_list_idx << " bboxes\n";

	etime = (double)(tt2.tv_sec - tt1.tv_sec) * 1000.0 + (double)(tt2.tv_nsec - tt1.tv_nsec) / 10000000.0;
	LOG(INFO) << "total time: " << etime << " ms, avg time: " << etime / (double)bbox_list_idx << " ms\n";
}


