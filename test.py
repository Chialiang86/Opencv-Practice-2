

def rgb_callback(self,data):
        try:
            img = self.br.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # ret, corners = cv2.findChessboardCorners(gray, (x_num,y_num),None)
        ret, corners = cv2.findChessboardCorners(img, (x_num,y_num))
        cv2.imshow('img',img)
        cv2.waitKey(5)
        if ret == True:
            cv2.cornerSubPix(gray,corners,(5,5),(-1,-1),criteria)
            tempimg = img.copy()
            cv2.drawChessboardCorners(tempimg, (x_num,y_num), corners,ret)
            # ret, rvec, tvec = cv2.solvePnP(objpoints, corners, mtx, dist, flags = cv2.CV_EPNP)
            rvec, tvec, inliers = cv2.solvePnPRansac(objpoints, corners, rgb_mtx, rgb_dist)
            print("rvecs:")
            print(rvec)
            print("tvecs:")
            print(tvec)
            # project 3D points to image plane
            imgpts, jac = cv2.projectPoints(axis, rvec, tvec, rgb_mtx, rgb_dist)

            imgpts = np.int32(imgpts).reshape(-1,2)

            cv2.line(tempimg, tuple(imgpts[0]), tuple(imgpts[1]),[255,0,0],4)  #BGR
            cv2.line(tempimg, tuple(imgpts[0]), tuple(imgpts[2]),[0,255,0],4)
            cv2.line(tempimg, tuple(imgpts[0]), tuple(imgpts[3]),[0,0,255],4)

            cv2.imshow('img',tempimg)
            cv2.waitKey(5)