// Derived from AMReX (c) 2024 The Regents of the University of California.
// Redistribution terms: reference/pressure2d/AMReX-LICENSE; source map: reference/engine/.
#include "AmrField2D.h"
#include <limits>

namespace fluid {

std::vector<AmrBox2D> splitAmrBoxes(const std::vector<AmrBox2D>& boxes,int maximum) {
    std::vector<AmrBox2D> result;
    for(const auto& box:boxes) {
        std::array<std::vector<std::pair<int,int>>,2> spans;
        for(int d=0;d<2;++d) {
            int length=box.length(d), chunk=maximum, ratio=1, blocks=1, size=length, extra=0;
            if(length>chunk) {
                while(chunk%2==0 && length%2==0){ratio*=2;chunk/=2;length/=2;}
                blocks=(length+chunk-1)/chunk;size=length/blocks;extra=length-size*blocks;
            }
            for(int i=0;i<blocks;++i) spans[d].emplace_back(box.smallEnd(d)+(i*size+std::min(i,extra))*ratio,(size+(i<extra))*ratio);
        }
        for(auto [y,h]:spans[1]) for(auto [x,w]:spans[0]) result.push_back({x,y,w,h});
    }
    return result;
}

namespace {
// Berger-Rigoutsos FindCut/new_chop, restricted to the two in-plane histograms.
std::pair<int,int> findCut(const std::vector<int>& hist,int lo) {
    const int length=int(hist.size()), mid=length/2;
    if(length<=1)return {lo,3};
    int cut=-1, status=3;
    for(int i=0;i<length;++i) if(hist[i]==0) {
        status=0;
        if(std::abs(cut-mid)>std::abs(i-mid)) {cut=i;if(i>mid)break;}
    }
    if(status==0)return {lo+cut,status};
    std::vector<int> derivative(length,0);
    for(int i=1;i<length-1;++i)derivative[i]=hist[i+1]-2*hist[i]+hist[i-1];
    int largest=-1;
    for(int i=2;i<length-2;++i) {
        const int change=std::abs(derivative[i-1]-derivative[i]);
        if(derivative[i-1]*derivative[i]<0 && change>=largest) {
            if(change>largest){status=1;cut=i;largest=change;}
            else if(std::abs(i-mid)<std::abs(cut-mid))cut=i;
        }
    }
    if(largest<=2){cut=mid;status=2;}
    return {lo+cut,status};
}
}

std::vector<AmrBox2D> clusterAmrTags(std::vector<std::array<int,2>> points,double efficiency) {
    struct Cluster { int first, count; AmrBox2D box; };
    auto bounds=[&](int first,int count) {
        int x=std::numeric_limits<int>::max(),y=x,xx=std::numeric_limits<int>::min(),yy=xx;
        for(int i=first;i<first+count;++i){x=std::min(x,points[i][0]);y=std::min(y,points[i][1]);xx=std::max(xx,points[i][0]);yy=std::max(yy,points[i][1]);}
        return AmrBox2D{x,y,xx-x+1,yy-y+1};
    };
    if(points.empty())return {};
    auto fraction=[](Cluster c){return double(c.count)/(c.box.width*c.box.height);};
    std::vector<Cluster> clusters{{0,int(points.size()),bounds(0,int(points.size()))}};
    for(int i=0;i<int(clusters.size());) {
        const auto old=clusters[i];
        if(fraction(old)>=efficiency){++i;continue;}
        std::array<std::vector<int>,2> hist{std::vector<int>(old.box.width),std::vector<int>(old.box.height)};
        for(int p=old.first;p<old.first+old.count;++p)for(int d=0;d<2;++d)++hist[d][points[p][d]-old.box.smallEnd(d)];
        int invalid=-1;
        for(int attempt=0;attempt<2;++attempt) {
            std::array<std::pair<int,int>,2> cuts;
            for(int d=0;d<2;++d)cuts[d]=d==invalid?std::pair{0,3}:findCut(hist[d],old.box.smallEnd(d));
            const int status=std::min(cuts[0].second,cuts[1].second);
            int direction=-1,balance=-1;
            for(int d=0;d<2;++d)if(cuts[d].second==status) {
                const int length=std::min(cuts[d].first-old.box.smallEnd(d),old.box.bigEnd(d)-cuts[d].first);
                if(length>=balance){direction=d;balance=length;}
            }
            auto begin=points.begin()+old.first, end=begin+old.count;
            auto middle=std::partition(begin,end,[&](const auto& p){return p[direction]<cuts[direction].first;});
            const int low=int(middle-begin);
            Cluster below{old.first,low,bounds(old.first,low)}, above{old.first+low,old.count-low,bounds(old.first+low,old.count-low)};
            if(fraction(below)>fraction(old)||fraction(above)>fraction(old)||attempt>0) {
                clusters[i]=below;clusters.push_back(above);break;
            }
            invalid=direction;
        }
    }
    std::vector<AmrBox2D> result;
    for(const auto& c:clusters)result.push_back(c.box);
    return result;
}

} // namespace fluid
