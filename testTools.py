# -*- coding: utf-8 -*-
"""
@AUTHORS: Chris Lloyd, Tom Gunter, Mike Osborne

IN NO EVENT SHALL THE AUTHORS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, 
SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING
OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF THE AUTHORS HAVE
BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

THE AUTHORS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED 
HEREUNDER IS PROVIDED "AS IS". THE AUTHORS HAVE NO OBLIGATION TO PROVIDE
MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
"""

import numpy as np

def unpack( grads ):
    
    if type( grads ) is list:
        
        # get the first set of derivative in the list
        g = grads[0]
        # find the number of elements in this derivative
        gs = g.size
        # create an array with that many elements by number of deriviatives
        fl = np.zeros( (len(grads),gs) )
        # put the first derivative in the array
        fl[0,:] = g.flatten()
        for i in range(1,len(grads)):
            # for each of the remaining gradients, flatten and put in array
            g = grads[i]
            fl[i,:] = g.flatten()
        # flatten the array of flatten arrays
        fl = fl.flatten()
        # force fl to be column vector
        fl = fl.reshape((-1,1))
        # return column vector of flattened 
        return fl
        
    elif type( grads ) is np.ndarray:
        grads = grads.flatten();
        grads = grads.reshape((-1,1))
        return grads
        
    else:
        
        return []
            
def checkgrad( f, df, x, e=1e-5, thresh=1e-5, disp=False, useAssert=False ):

    dy = unpack( df( x ) )
    y = f( x )    
    
    nrows = x.size 
    ncols = y.size
    
    dh = np.zeros( (nrows,ncols) )
    for i in range( x.size ):
        
        x[i] = x[i] - e
        y1 = f( x )
        x[i] = x[i] + 2 * e
        y2 = f( x )
        x[i] = x[i] - e
        
        g = (y2 - y1) / (2 * e)
        dh[i,:] = g.flatten() 
    
    dh = dh.flatten()
    dh = dh.reshape((-1,1))    
    
    if disp:
        pnt = np.concatenate( (dy,dh), axis=1 )
        print '  Analytic    Numerical'
        print( pnt )
    
    numer = np.linalg.norm( dh - dy )
    denom = np.linalg.norm( dh + dy )
    
    if  numer != 0 and denom != 0:
        if useAssert:
            assert( numer / denom < thresh )
        elif numer / denom > thresh:
            print("ERROR Test Failed")
    
    
# If used as a standalone module, then run the tests
if __name__ == "__main__":    
    
    def testf( x ):
        c = np.ndarray( (2,2) )
        c[0,0] = 3
        c[1,0] = 2
        c[0,1] = 6
        c[1,1] = 1
        y = np.sin( c * x[0] ) + np.cos( c * x[1] )
        return y

    def testdf( x ):
        c = np.ndarray( (2,2) )
        c[0,0] = 3
        c[1,0] = 2
        c[0,1] = 6
        c[1,1] = 1
        dy = [0,0]        
        dy[0] =  np.cos( c * x[0] ) * c
        dy[1] = -np.sin( c * x[1] ) * c
        return dy 
    
    x = np.zeros( (2,1) )
    x[0] = 1.4
    x[1] = 3.2
    checkgrad( testf, testdf, x, disp=True )
    