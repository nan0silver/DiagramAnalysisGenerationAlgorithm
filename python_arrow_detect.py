import cv2
import numpy as np
import math
from sys import stdin
import requests, json


def preprocess(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)
    # cv2.imshow("img_blur", img_blur)
    # cv2.waitKey(0)

    img_canny = cv2.Canny(img_blur, 50, 50)
    # cv2.imshow("img_canny", img_canny)
    # cv2.waitKey(0)

    kernel = np.ones((3, 3))
    img_dilate = cv2.dilate(img_canny, kernel, iterations=2)
    img_erode = cv2.erode(img_dilate, kernel, iterations=2)
    cv2.imshow("img_erode", img_erode)
    cv2.waitKey(0)

    return img_erode

def find_tip(points, convex_hull):
    length = len(points) #모든 꼭짓점 갯수(화살표의 경우 6 or 7)
    print("length = %d" % length)
    print(points)
    indices = np.setdiff1d(range(length), convex_hull)
    #첫번째 배열 x로 부터 두번째 배열 y를 뺀 차집합을 반환
    print(convex_hull)
    print(indices)
    for i in range(2):
        j = indices[i] + 2
        print("j : %d" % j)
        if j > length - 1:
            j = length - j
        if np.all(points[j] == points[indices[i - 1] - 2]):
            #print(j)
            return j
        

def find_endpoint(points, convex_hull, tip): #points는 화살표의 모든 꼭짓점 리스트
    length = len(points) #모든 꼭짓점 갯수
    #if length == 7:
    temp = tip+3
    if temp > length-1:
        return tuple(points[temp-length])
    return tuple(points[temp])


def cal_distance(arrow_point, rec_point):
    ax = arrow_point[0]
    ay = arrow_point[1]
    rx = rec_point[0]
    ry = rec_point[1]

    result = math.sqrt( math.pow(ax - rx, 2) + math.pow(ay - ry, 2))
    return result

def arrow_startTo_rec(arrow_start, rec_list):
    temp_list = []
    for rec in rec_list:
        temp_val = cal_distance(arrow_start, rec)
        temp_list.append(temp_val)
        #rec[2]는 rec의 index
        #print("val : %d, rec[2] : %d" % (temp_val, rec[2]))
        #print(rec)
    temp_min = temp_list.index(min(temp_list))

    return rec_list[temp_min][2] #가장 거리가 가까운 사각형의 인덱스


#화살표의 시작점
#rec_center를 사용해야 마름모도 정확히 감지 가능
def detect_nearest_box_start(arrow_coordinate, arrow_vector, rec_list): 
    temp_list = []
    abs_x = abs(arrow_vector[0])
    abs_y = abs(arrow_vector[1])
    #print("abs X : %d, abs Y : %d" % (abs_x, abs_y))
    if abs_x < abs_y:
        #수직방향
        if arrow_vector[1] > 0:
            #아래쪽 방향
            for rec1 in rec_list:
                if rec1[1] >= arrow_coordinate[1]:
                    #화살표의 y좌표값보다 사각형 센터의 y좌표값이 큰 경우만
                    temp_val = cal_distance(arrow_coordinate, rec1)
                    temp_list.append([temp_val, rec1[2]])
        else:
            #위쪽 방향
            for rec1 in rec_list:
                if rec1[1] <= arrow_coordinate[1]:
                    #화살표의 y좌표값보다 사각형 센터의 y좌표값이 작은 경우만
                    temp_val = cal_distance(arrow_coordinate, rec1)
                    temp_list.append([temp_val, rec1[2]])     
    else:
        #수평방향
        if arrow_vector[0] > 0: 
            # --> 방향
            for rec2 in rec_list:
                if rec2[0] >= arrow_coordinate[0]:
                    #화살표의 x좌표값보다 사각형 센터의 x좌표값이 큰 경우만
                    print(rec2[2])
                    temp_val = cal_distance(arrow_coordinate, rec2)
                    temp_list.append([temp_val, rec2[2]])
        else:
            # <-- 방향
            for rec2 in rec_list:
                if rec2[0] <= arrow_coordinate[0]:
                    #화살표의 x좌표값보다 사각형 센터의 x좌표값이 작은 경우만
                    print(rec2[2])
                    temp_val = cal_distance(arrow_coordinate, rec2)
                    temp_list.append([temp_val, rec2[2]])

    temp_list.sort()

    return temp_list[0][1] #가장 거리가 가까운 사각형의 인덱스

def detect_nearest_box_end(arrow_coordinate, arrow_vector, rec_list): 
    temp_list = []
    abs_x = abs(arrow_vector[0])
    abs_y = abs(arrow_vector[1])
    print("abs X : %d, abs Y : %d" % (abs_x, abs_y))
    if abs_x < abs_y:
        #수직방향
        print("수직")
        if arrow_vector[1] > 0:
            #아래쪽 방향
            for rec1 in rec_list:
                if rec1[1] <= arrow_coordinate[1]:
                    #화살표의 y좌표값보다 사각형 센터의 y좌표값이 작은 경우만
                    temp_val = cal_distance(arrow_coordinate, rec1)
                    temp_list.append([temp_val, rec1[2]])  
        else:
            #위쪽 방향   
            for rec1 in rec_list:
                if rec1[1] >= arrow_coordinate[1]:
                    #화살표의 y좌표값보다 사각형 센터의 y좌표값이 큰 경우만
                    temp_val = cal_distance(arrow_coordinate, rec1)
                    temp_list.append([temp_val, rec1[2]])  
    else:
        #수평방향
        print("수평")
        if arrow_vector[0] > 0: 
            # --> 방향
            for rec2 in rec_list:
                if rec2[0] <= arrow_coordinate[0]:
                    #화살표의 x좌표값보다 사각형 센터의 x좌표값이 작은 경우만
                    print(rec2[2])
                    temp_val = cal_distance(arrow_coordinate, rec2)
                    temp_list.append([temp_val, rec2[2]])
        else:
            # <-- 방향
            for rec2 in rec_list:
                if rec2[0] >= arrow_coordinate[0]:
                    #화살표의 x좌표값보다 사각형 센터의 x좌표값이 큰 경우만
                    print(rec2[2])
                    temp_val = cal_distance(arrow_coordinate, rec2)
                    temp_list.append([temp_val, rec2[2]])

    print(arrow_coordinate)
    print(temp_list)
    temp_list.sort()
    print(temp_list[0][1])
    return temp_list[0][1] #가장 거리가 가까운 사각형의 인덱스


def draw_rec_by_idx(idx, rec_list, img):
    #idx는 rec의 인덱스
    cv2.rectangle(img, (rec_list[idx*4][0], rec_list[idx*4][1]), (
    rec_list[idx*4+3][0], rec_list[idx*4+3][1]), (255, 0, 0), 5)



def requestOCR(image_url):
    url = "https://tc61wu7n11.apigw.ntruss.com/custom/v1/20088/38629e69fbf3c3640428ce080d3b72bdfd5c15b29f5e9472460db30cf365cf48/general"
    header = {'Content-Type': 'application/json', 'X-OCR-SECRET':'키 넣기'}
    data = {"images": [
            {
                "format": "png",
                "name": "medium",
                "url": image_url
            }
            ],
            "lang": "ko",
            "requestId": "string",
            "resultType": "string",
            "timestamp": 0,
            "version": "V1"}

    try:
        res = requests.post(url, headers=header, data=json.dumps(data))
    except requests.exceptions.Timeout as errd:
        print("Timeout Error : ", errd)
        
    except requests.exceptions.ConnectionError as errc:
        print("Error Connecting : ", errc)
        
    except requests.exceptions.HTTPError as errb:
        print("Http Error : ", errb)

    # Any Error except upper exception
    except requests.exceptions.RequestException as erra:
        print("AnyException : ", erra)

    print("********res**********")
    print(res)




image_index = "1"
image_name = "compare_image"+image_index+".png"
#test_diagram
#compare_image

img = cv2.imread("image/"+image_name)

contours, hierarchy = cv2.findContours(preprocess(img), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#밑에껀 컨투어의 꼭짓점 좌표만

idx =0
arrow_count=0
rec_count=0
rec_list = []
rec_sort = []
arrow_list = []
arrow_vector = []
arrow_vector_original = []
rec_XYWH = []
rec_center = []
isDiagramVertical = False

img_size = img.shape
empty_win = np.zeros(img_size, np.uint8)

for cnt in contours:
    cv2.drawContours(img, [cnt],-1,  (255, 0, 0), 4)
    peri = cv2.arcLength(cnt, True)
    #외곽선 길이 반환
    approx = cv2.approxPolyDP(cnt, 0.025 * peri, True)
    #요철이 있는 부분은 무시하고 컨투어를 계산
    #근사치 정확도의 값이 낮을 수록, 근사를 더 적게해 원본 윤곽과 유사해짐
    #2번째 파라미터가 근사치 정확도
    #너무 작으면(0.015) 짧은 화살표가 탐지되지 않고, 너무 크면(0.025) 긴 화살표가 탐지되지 않는다.
    hull = cv2.convexHull(approx, returnPoints=False)
    sides = len(hull)

    for app in approx:
        cv2.circle(img, tuple(app[0]), 3, (0, 0, 255), -1)

    print("[%d] sides : %d len(approx) : %d" % (idx, sides, len(approx)))
    idx=idx+1


    if 6 > sides > 3 and sides + 2 == len(approx): #화살표의 경우
        points = approx[:,0,:]
        arrow_tip = find_tip(points, hull.squeeze())
        arrow_tip_tuple = tuple(points[arrow_tip])
        if arrow_tip != None:
            
            arrow_endpoint = find_endpoint(points, hull.squeeze(), arrow_tip)
            cv2.drawContours(img, [cnt], -1, (0, 255, 0), 3) #g
            cv2.circle(img, arrow_tip_tuple, 3, (0, 0, 255), cv2.FILLED) #r
            cv2.circle(empty_win, arrow_tip_tuple, 3, (0, 0, 255), cv2.FILLED) #r
            cv2.circle(img, arrow_endpoint, 3, (255, 0, 0), cv2.FILLED) #b
            cv2.circle(empty_win, arrow_endpoint, 3, (255, 255, 0), cv2.FILLED) #b

            arrow_list.append((arrow_tip_tuple, arrow_endpoint, arrow_count, False)) 
            #arrow 인덱스는 0부터 시작
            #마지막은 arrow flag

            arrow_vector_x = arrow_tip_tuple[0] - arrow_endpoint[0]
            arrow_vector_y = arrow_tip_tuple[1] - arrow_endpoint[1]
            arrow_vector_original.append([arrow_vector_x, arrow_vector_y])

            temp_vector = np.array([arrow_vector_x, arrow_vector_y])
            temp_unit_vector = temp_vector / np.linalg.norm(temp_vector)
            arrow_vector.append([temp_unit_vector[0], temp_unit_vector[1]])

            arrow_count=arrow_count+1
            print("arrow_tip index : %d" % arrow_tip)


    # elif sides == 4:   #사각형의 경우
    #     x, y, w, h = cv2.boundingRect(cnt)
    #     rec_list.append([x, y, rec_count]) #사각형의 인덱스는 0부터 시작
    #     rec_list.append([x, y+h, rec_count])
    #     rec_list.append([x+w, y, rec_count])
    #     rec_list.append([x+w, y+h, rec_count])

    #     rec_sort.append((x, y, rec_count))

    #     rec_center.append([(2*x+w)/2, (2*y+h)/2, rec_count])

    #     rec_XYWH.append([x, y, w, h, rec_count])
    #     rec_count=rec_count+1
    #     print("[사각형] [%d] (x, y, w, h) : (%d, %d, %d, %d)" % (rec_count-1, x, y , w, h))
    
    else:
        approx = cv2.approxPolyDP(cnt, 0.015 * peri, True)
        #요철이 있는 부분은 무시하고 컨투어를 계산
        #근사치 정확도의 값이 낮을 수록, 근사를 더 적게해 원본 윤곽과 유사해짐
        #2번째 파라미터가 근사치 정확도
        #너무 작으면(0.015) 짧은 화살표가 탐지되지 않고, 너무 크면(0.025) 긴 화살표가 탐지되지 않는다.
        hull = cv2.convexHull(approx, returnPoints=False)
        sides = len(hull)

        for app in approx:
            cv2.circle(img, tuple(app[0]), 3, (0, 255, 0), -1)
        
        # print("[%d] sides : %d len(approx) : %d" % (idx, sides, len(approx)))
        # idx=idx+1

        if 6 > sides > 3 and sides + 2 == len(approx): #화살표의 경우
            points = approx[:,0,:]
            arrow_tip = find_tip(points, hull.squeeze())
            arrow_tip_tuple = tuple(points[arrow_tip])
            if arrow_tip != None:
                
                arrow_endpoint = find_endpoint(points, hull.squeeze(), arrow_tip)
                cv2.drawContours(img, [cnt], -1, (0, 255, 0), 3) #g
                cv2.circle(img, arrow_tip_tuple, 3, (0, 0, 255), cv2.FILLED) #r
                cv2.circle(empty_win, arrow_tip_tuple, 3, (0, 0, 255), cv2.FILLED) #r
                cv2.circle(img, arrow_endpoint, 3, (255, 0, 0), cv2.FILLED) #b
                cv2.circle(empty_win, arrow_endpoint, 3, (255, 255, 0), cv2.FILLED) #b

                arrow_list.append((arrow_tip_tuple, arrow_endpoint, arrow_count, False)) 
                #arrow 인덱스는 0부터 시작
                #마지막은 arrow flag

                arrow_vector_x = arrow_tip_tuple[0] - arrow_endpoint[0]
                arrow_vector_y = arrow_tip_tuple[1] - arrow_endpoint[1]
                arrow_vector_original.append([arrow_vector_x, arrow_vector_y])

                temp_vector = np.array([arrow_vector_x, arrow_vector_y])
                temp_unit_vector = temp_vector / np.linalg.norm(temp_vector)
                arrow_vector.append([temp_unit_vector[0], temp_unit_vector[1]])

                arrow_count=arrow_count+1
                print("arrow_tip index : %d" % arrow_tip)

        elif sides == 4:   #사각형의 경우
            x, y, w, h = cv2.boundingRect(cnt)
            rec_list.append([x, y, rec_count]) #사각형의 인덱스는 0부터 시작
            rec_list.append([x, y+h, rec_count])
            rec_list.append([x+w, y, rec_count])
            rec_list.append([x+w, y+h, rec_count])

            rec_sort.append((x, y, rec_count))

            rec_center.append([(2*x+w)/2, (2*y+h)/2, rec_count])

            rec_XYWH.append([x, y, w, h, rec_count])
            rec_count=rec_count+1
            print("[사각형] [%d] (x, y, w, h) : (%d, %d, %d, %d)" % (rec_count-1, x, y , w, h))



# 다이어그램의 모든 화살표들의 벡터 계산
diagram_vector_x = 0
diagram_vector_y = 0
for av in arrow_vector:
    diagram_vector_x = diagram_vector_x+av[0]
    diagram_vector_y = diagram_vector_y+av[1]

print("diagram X : %d, diagram Y : %d" % (diagram_vector_x, diagram_vector_y))

if diagram_vector_x > diagram_vector_y:
    print("This diagram is HORIZONTAL")
    #search closest box with Y line 
    #y축에 가장 가까운, x좌표값이 가장 작은 박스 찾기
else :
    isDiagramVertical = True
    print("This diagram is VERTICAL")
    #y값 먼저, x값 다음으로 정렬
        

#여기서 사각형 인덱스 정렬
print("\n\n################")
print("rec sort before")
print(rec_sort)

if isDiagramVertical:
    rec_sort.sort(key=lambda x: (x[1], x[0]))
else:
    rec_sort.sort()
print("\nrec sort after")
print(rec_sort)

for rl in rec_list:
    for rsidx, rs in enumerate(rec_sort):
        if rs[2] == rl[2]:
            rl.append(rsidx)

for rl in rec_list:
    del rl[2]

for rxidx, rx in enumerate(rec_XYWH):
    rx[4] = rec_list[rxidx*4][2]

for rcinx, rc in enumerate(rec_center):
    rc[2] = rec_list[rcinx*4][2]

print("")
print("********************************")
print(rec_XYWH)
print("+++++++++++++++++++++++++++++++")
print(rec_list)
print("********************************")
print(rec_center)
print("")


# 여기까지 각 도형에 대해 사각형인지, 화살표인지, 화살표면 시작점 끝점 어디인지, 
# 사각형이면 꼭짓점 4개의 좌표는 무엇인지 다 알아냄
# 이제는 사각형과 화살표 사이 관계성을 알아내야함

# (0,0)에서 가장 가까운 사각형 -> 화살표
# 화살표 -> 사각형 -> 화살표 순으로 알아가야함
# 각 점마다 최단거리를 가지는 점을 가진 도형을 알아내야함
distance_list = []
isThisArrow = True
rec_to_arrow_list = [0 for i in range(rec_count)]
rec_from_arrow_list = [0 for i in range(rec_count)]


# 모든 화살표에 대해 시작, 끝점 모두 연결되는 하나의 박스를 찾아둔다.
arr_connect_list = []

print("arrow list")
print(arrow_list)
print("")
print("arrow vector")
print(arrow_vector)
print("")
print("arrow vector original")
print(arrow_vector_original)
print("")


# cv2.imshow("rec Window", empty_win)
# cv2.imshow("Image", img)
# cv2.waitKey(0)

for arr in arrow_list:
    startP = arr[0]
    endP = arr[1]
    arr_idx = arr[2]

    startRec = detect_nearest_box_start(startP, arrow_vector_original[arr_idx], rec_center)
    endRec = detect_nearest_box_end(endP, arrow_vector_original[arr_idx], rec_center)

    #draw_rec_by_idx(endRec, rec_list, img)

    arr_connect_list.append((arr_idx, startRec, endRec))
    #(화살표 인덱스, 시작점과 연결되는 박스 인덱스, 끝점과 연결되는 박스 인덱스)

print("arr_connect_list")
print(arr_connect_list)


adj = [ [] for _ in range(rec_count)]

for con in arr_connect_list:
    adj[con[2]].append(con[1])

print("adj")
print(adj)


idx2 = "idx"
for rec in range(rec_count):
    draw_rec_by_idx(rec, rec_list, empty_win)
    #print(rec)
    rec_num = rec_list[rec*4][2]
    cv2.putText(empty_win, str(rec_num), (rec_list[rec*4][0], rec_list[rec*4][1]-5), cv2.FONT_HERSHEY_SIMPLEX, 
               1, (255, 0, 0), 3)
    cv2.circle(empty_win, (int(rec_center[rec][0]), int(rec_center[rec][1])), 3, (0, 0, 255), cv2.FILLED)
    #cv2.circle(empty_win, (rec_list[rec*4][0], rec_list[rec*4][1]-5), (255, 0, 0), 3)
    idx=idx+1

cv2.imshow("rec Window", empty_win)
cv2.imshow("Image", img)
cv2.waitKey(0)

#requestOCR("이미지 주소")

requestResult = [ [] for _ in range(rec_count)]

open_path = "jsonFile/compare_image"+image_index+".json"
#test_diagram
#compare_image

with open(open_path) as json_file:
    json_data = json.load(json_file)

    json_fields = json_data['images'][0]['fields']

    for obj in json_fields:
        json_vet0 = obj['boundingPoly']['vertices'][0] #좌표 리스트
        json_vet2 = obj['boundingPoly']['vertices'][2]
        json_test = obj['inferText']


        for k in rec_XYWH:
            if ((k[0] <= json_vet0['x'] <= (k[0]+k[2])) and (k[1] <= json_vet0['y'] <= (k[1]+k[3]))):
                requestResult[k[4]].append(json_test)


print(requestResult)
print("rec text : %d" % rec_list[0][1])
print("arrow_count : %d" % arrow_count)
print("rec_count : %d" % rec_count)

requestResult_join = []
for rr in requestResult:
    requestResult_join.append(" ".join(rr))

print(requestResult_join)

#adj = [[1], [2, 3], [], [2, 4], []]
sentence_result = ''

for i in range(rec_count) :
    for adj_temp in adj[i]:
        sentence_result=sentence_result+requestResult_join[i]
        sentence_result=sentence_result+' is connected with '
        sentence_result=sentence_result+requestResult_join[adj_temp]
        sentence_result=sentence_result+". \n"

print(" ")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print(sentence_result)
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print(" ")
        

# cv2.imshow("Image", img)
# cv2.waitKey(0)