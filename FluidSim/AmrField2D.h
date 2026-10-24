#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace fluid {

struct AmrBox2D {
    int x, y, width, height;
    int smallEnd(int d) const { return d ? y : x; }
    int bigEnd(int d) const { return smallEnd(d) + length(d) - 1; }
    int length(int d) const { return d ? height : width; }
    bool contains(int i, int j) const { return i >= x && j >= y && i < x+width && j < y+height; }
    AmrBox2D grown(int n) const { return {x-n,y-n,width+2*n,height+2*n}; }
    AmrBox2D faces(int d) const { return {x,y,width+(d==0),height+(d==1)}; }
    AmrBox2D coarse(int ratio) const { return {x/ratio,y/ratio,width/ratio,height/ratio}; }
    bool operator==(const AmrBox2D&) const = default;
};

inline AmrBox2D intersect(AmrBox2D a, AmrBox2D b) {
    const int x=std::max(a.x,b.x), y=std::max(a.y,b.y);
    return {x,y,std::max(0,std::min(a.x+a.width,b.x+b.width)-x),std::max(0,std::min(a.y+a.height,b.y+b.height)-y)};
}

template<class Function> inline void forCells(AmrBox2D box, Function&& function) {
    for(int y=box.y;y<box.y+box.height;++y)
        for(int x=box.x;x<box.x+box.width;++x) function(x,y);
}
template<class Function> inline void forCells(AmrBox2D box, int components, Function&& function) {
    for(int n=0;n<components;++n) forCells(box,[&](int x,int y){function(x,y,n);});
}

// Component-major patch storage. Ghosts belong to the patch, never a hidden fine grid.
template<class T> struct AmrArray2D {
    T* values = nullptr;
    int x=0, y=0, stride=0, plane=0;
    int index(int i,int j) const { return (j-y)*stride+i-x; }
    T& operator()(int i,int j,int n=0) const { return values[index(i,j)+n*plane]; }
};

template<class T> class AmrField2D {
public:
    struct Patch { AmrBox2D box; int ghosts, plane; std::vector<T> values; };
    void define(const std::vector<AmrBox2D>& boxes,int components,int ghosts) {
        count=components; data.clear(); data.reserve(boxes.size());
        for(const auto& box:boxes) {
            const int plane=(box.width+2*ghosts)*(box.height+2*ghosts);
            data.push_back({box,ghosts,plane,std::vector<T>(size_t(plane)*components)});
        }
    }
    int size() const { return int(data.size()); }
    int nComp() const { return count; }
    AmrBox2D box(int p) const { return data[p].box; }
    AmrArray2D<T> array(int p) { auto& a=data[p];return {a.values.data(),a.box.x-a.ghosts,a.box.y-a.ghosts,a.box.width+2*a.ghosts,a.plane}; }
    AmrArray2D<const T> const_array(int p) const { const auto& a=data[p];return {a.values.data(),a.box.x-a.ghosts,a.box.y-a.ghosts,a.box.width+2*a.ghosts,a.plane}; }
    void setVal(T value) { for(auto& p:data) std::fill(p.values.begin(),p.values.end(),value); }
    void multiply(T value,int first,int components) {
#pragma omp parallel for
        for(int p=0;p<size();++p) {
            const auto q=array(p);
            forCells(box(p),components,[&](int x,int y,int n){q(x,y,first+n)*=value;});
        }
    }
    double norm0(int n) const {
        double result=0.;
#pragma omp parallel for reduction(max:result)
        for(int p=0;p<size();++p) {
            const auto q=const_array(p);
            forCells(box(p),[&](int x,int y){result=std::max(result,std::abs(double(q(x,y,n))));});
        }
        return result;
    }
    void copyValid(const AmrField2D& source) {
#pragma omp parallel for
        for(int p=0;p<size();++p) {
            const auto q=array(p);const auto in=source.const_array(p);
            for(int n=0;n<count;++n) for(int y=box(p).y;y<=box(p).bigEnd(1);++y)
                std::copy_n(&in(box(p).x,y,n),box(p).width,&q(box(p).x,y,n));
        }
    }
private:
    int count=0;
    std::vector<Patch> data;
};

struct AmrGeometry2D {
    int width, height;
    double cell_size;
    AmrBox2D Domain() const { return {0,0,width,height}; }
    double CellSize(int) const { return cell_size; }
};

// AMReX BoxList::maxSize: preserve factors of two while balancing patch lengths.
std::vector<AmrBox2D> splitAmrBoxes(const std::vector<AmrBox2D>& boxes,int maximum);
std::vector<AmrBox2D> clusterAmrTags(std::vector<std::array<int,2>> points,double efficiency);

} // namespace fluid
