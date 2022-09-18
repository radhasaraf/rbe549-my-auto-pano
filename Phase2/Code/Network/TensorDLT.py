import torch

class TensorDlt(nn.Module):
    def forward(x,x_p):
        ## TODO wont work for minibatch size of 1 yet
        B_size = x.shape[0]

        # intialize and A and b matrices
        A = torch.zeros(size=(B_size,8,8),dtype=torch.float64)
        b = torch.zeros(size=(B_size,8),dtype=torch.float64)

        idx_x = torch.tensor([0,2,4,6])
        idx_y = torch.tensor([1,3,5,7])
        zeros = torch.zeros(size=(B_size,4))
        ones = torch.ones(size=(B_size,4))

        
        u_i,v_i = x.reshape(shape=(-1,4,2)).transpose(1,2).swapaxes(0,1)
        u_i_p,v_i_p = x_p.reshape(shape=(-1,4,2)).transpose(1,2).swapaxes(0,1)
     
        A[:,idx_x] = torch.stack([zeros,zeros,zeros,-u_i,-v_i,-ones,v_i_p*u_i,v_i_p*v_i],dim=2)
        A[:,idx_y] = torch.stack([u_i,v_i,ones,zeros,zeros,zeros,-u_i_p*u_i,-u_i_p*v_i],dim=2)
        b[:,idx_x] = -v_i_p
        b[:,idx_y] = u_i_p

        b = b.reshape(B_size,8,1)
        ret = torch.linalg.pinv(A) @ b
        ones_2 = torch.ones(size=(B_size,1,1))
        ret = torch.cat([ret,ones_2],dim=1)
        ret = ret.reshape(shape=(-1,3,3))
        return ret

def test_tensorDLT():
    test_L = torch.tensor([
                            [
                                [-23,   -18, -29 , 29 , -5 , -22 , 5 ,  32],
                                [-213, -181, -219, 219, -15, -212, 15, 312],
                                [-223, -281, -229, 229, -25, -222, 25, 322],
                                [-233, -381, -239, 239, -35, -232, 35, 322],
                            ],
                            [
                                [-1  , -8  , -91 , 9 , -65  , -4 , -5,  37],
                                [-233, -118, -229, 291, -513, -21, 52, -20],
                                [-243, -128, -239, 292, -523, -22, 62, -30],
                                [-253, -138, -249, 293, -533, -23, 72, -40]
                            ]
                        ],dtype=torch.float64)
    actual_H4pt_list = [
                            [
                                [0,   0, 128 , 0 , 0 , 128 , 128 ,  128],
                                [-213, -181, -219, 219, -15, -212, 15, 312],
                            ],
                            [
                                [-23, -18, 99, 29, -5, 106, 133, 160],
                                [-233, -118, -229, 291, -513, -21, 52, -20],
                            ]

                    ]

    import numpy as np
    H4pt_src = np.array(actual_H4pt_list[0][0]).reshape(4,2)
    print(f'src:{H4pt_src}')
    H4pt_dst = np.array(actual_H4pt_list[1][0]).reshape(4,2)
    print(f'dst:{H4pt_dst}')

    import cv2
    H_cv2,_ = cv2.findHomography(H4pt_src,H4pt_dst)

    H4pt_torch = torch.tensor(actual_H4pt_list,dtype=torch.float64)

    # test_img = torch.zeros(size=(2,128,128))
    H_torch = tensorDLT(H4pt_torch[0],H4pt_torch[1])
    print(H_cv2)
    print(H_torch[0])

### Tript's version of TensorDLT
class TensorDLT(nn.Module):
    def init(self) -> None:
        '''
        H_4pt_X is H_4pt for batch X
        H4pt is the predicted H_4pt by the homography net, use it to calculate the corners in the predicted/ warped image
        C_a are the corner points of the patch in Image A or in this case the training image
        '''
        # self.H4pt = H4pt
        # self.C_a = C_a
        super().init()

    def tensorDLT(self, H_4pt_X, C_a):
        H = torch.tensor([])
        for H4pt in H_4pt_X:
            #corners are [x1 y1 x2 y2 x3 y3 x4 y4]
            Cb = C_a + H4pt
            A = []
            b = []
            for i in range(0,8,2): #since there are 4 corner pairs
                Ai = [[0, 0, 0, -C_a[i], -C_a[i+1], -1, Cb[i+1]C_a[i], Cb[i+1]C_a[i+1]]]
                Ai.append([C_a[i], C_a[i+1], 1, 0, 0, 0, -Cb[i]C_a[i], -Cb[i]C_a[i+1]])
                A.extend(Ai)

                bi = [-Cb[i+1],-Cb[i]]
                b.extend(bi)

            A = torch.tensor(A).to('cuda')
            b = torch.tensor(b).to('cuda')
            # h = inv(A) dot b
            print(A)
            h = torch.dot(torch.inverse(A), b)
            H = torch.cat(H,h.reshape(1,-1), axis=0)
        H = H[1:,:]
        print(H.shape)
        return H

    def forward(self,H_4pt_X, C_a):
        return self.tensorDLT(H_4pt_X, C_a)
