---
title: "OPTIMIZATION CLASS PROJECT TEAM WORK"
author: "RuiRui"
date: "October 28, 2015"
output: html_document
---
#Implementation of Line Search Algorithm By R

###difference:
- Change the logic of zoom function 
- Add tol to prevent from endless looping

```r
lineSearchAlgorithm=function(c1,c2,alpha1,alphaMax,obfn,obfnDeriv,mySelect,tol=10^(-7)){
    zoom=function(z1,z2){
        repeat{
            if(z2<tol)return(z2)
            tmp=(z1+z2)/2
            if(obfn(tmp)>obfn(0)+c1*tmp*obfnDeriv(0)){
                z2=tmp
                next
            }
            if(obfn(tmp)>=obfn(z1)){
                z2=tmp
                next
            }
            
            if(abs(obfnDeriv(tmp))<=-c2*obfnDeriv(0)){
                return(tmp)
            }
            if(obfnDeriv(tmp)<0){
                z1=tmp   
            }else{
                z2=tmp
            }
        }
    }
    alpha0=0
    repeat{
        if(obfn(alpha1)>obfn(0)+c1*alpha1*obfnDeriv(0)){
            return(zoom(alpha0,alpha1))
        }
        if(obfn(alpha1)>=obfn(alpha0)&(alpha0!=0)){
            return(zoom(alpha0,alpha1))
        }
        if(abs(obfnDeriv(alpha1))<=-c2*obfnDeriv(0)){
            return(alpha1)
        }
        if(obfnDeriv(alpha1)>=0){
            return(zoom(alpha0,alpha1))
        }
        alpha0=alpha1
        alpha1=mySelect(alpha1,alphaMax)
    }
}
```

An example of one step search is as follow:

```r
library(ggplot2)
theSolution=lineSearchAlgorithm(c1=10^(-4),c2=0.9,alpha1=10^(-4),alphaMax=1000,
                    obfn=function(x){
                        (x)^2-x  
                    },
                    obfnDeriv = function(x){
                        2*(x)-1
                    },
                    mySelect=function(m,M){
                        m*1.1
                    }
)
ggplot(data.frame(x=c(0,1)),aes(x,color="red",size=1))+stat_function(fun=function(x){x^2-x})+geom_vline(xintercept=theSolution)
```

![plot of chunk unnamed-chunk-2](figure/unnamed-chunk-2-1.png) 

Use the algorithm above to implement a simple one dimentional Search Algorithm

```r
simple_oneDimentionSearch=function(obfn,obfnDeriv,tol=10^(-7)){
    simple_search=function(obfn,obfnDeriv){
        lineSearchAlgorithm(c1=10^(-4),c2=0.9,alpha1=10^(-4),alphaMax=1000
                            ,obfn=obfn
                            ,obfnDeriv=obfnDeriv
                            ,mySelect=function(m,M){
                                m*1.1
                            })
    }
    shiftFunction=function(fn,shift,direction){
        newFn=function(x){
            fn(x*direction+shift)
        }
        return(newFn)
    }
    shiftDeriv=function(fn,shift,direction){
        newFn=function(x){
            direction*fn(x*direction+shift)
        }
        return(newFn)
    }
    
    thePoint=0
    repeat{
        if(obfnDeriv(thePoint)>=0){
            direction=-1
        }else{
            direction=1
        }  
        tmp=simple_search(shiftFunction(obfn,thePoint,direction=direction)
                          ,shiftDeriv(obfnDeriv,thePoint,direction=direction))
        
        thePoint=thePoint+tmp*direction
        if(abs(tmp)<tol) break
    }
    return(thePoint)
}
```

some Examples

```r
theSolution=simple_oneDimentionSearch(
    obfn=function(t){
        t^2-10*t+36
    },
    obfnDeriv=function(t){
        2*t-10
    }
)
ggplot(data.frame(x=c(0,10)),aes(x,color="red",size=1))+stat_function(fun=function(t){t^2-10*t+36})+geom_vline(xintercept=theSolution)
```

![plot of chunk unnamed-chunk-4](figure/unnamed-chunk-4-1.png) 




```r
theSolution=simple_oneDimentionSearch(
    obfn=function(t){
        -sin(t)
    },
    obfnDeriv=function(t){
        -cos(t)
    }
)
ggplot(data.frame(x=c(0,10)),aes(x,color="red",size=1))+stat_function(fun=function(t){-sin(t)})+geom_vline(xintercept=theSolution)
```

![plot of chunk unnamed-chunk-5](figure/unnamed-chunk-5-1.png) 


```r
theSolution=simple_oneDimentionSearch(
    obfn=function(t){
        exp(-t)+exp(t)
    },
    obfnDeriv=function(t){
        -exp(-t)+exp(t)
    }
)
ggplot(data.frame(x=c(-5,5)),aes(x,color="red",size=1))+stat_function(fun=function(t){exp(-t)+exp(t)})+geom_vline(xintercept=theSolution)
```

![plot of chunk unnamed-chunk-6](figure/unnamed-chunk-6-1.png) 


```r
theSolution=simple_oneDimentionSearch(
    obfn=function(x){
        cos(x)-0.5*x  
    },
    obfnDeriv = function(x){
        -sin(x)-0.5
    }
)
ggplot(data.frame(x=c(-5,10)),aes(x,color="red",size=1))+stat_function(fun=function(x){ cos(x)-0.5*x })+geom_vline(xintercept=theSolution)
```

![plot of chunk unnamed-chunk-7](figure/unnamed-chunk-7-1.png) 
