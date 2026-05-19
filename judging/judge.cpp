#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <stdexcept>
#include <algorithm>

static float fp16(float x) {
    uint32_t u; memcpy(&u, &x, 4);
    uint32_t s = u >> 31, e32 = (u >> 23) & 0xFF, m32 = u & 0x7FFFFF;
    if (e32 == 0xFF) { uint32_t r = (s<<31)|(0xFF<<23)|(m32?0x400000:0); float f; memcpy(&f,&r,4); return f; }
    int e16 = (int)e32 - 127 + 15;
    if (e16 >= 31) { uint32_t r=(s<<31)|(0xFF<<23); float f; memcpy(&f,&r,4); return f; }
    if (e16 <= 0) { float f=0.f; if(s) f=-0.f; return f; }
    uint32_t m16 = m32 >> 13, rb = (m32>>12)&1, st = m32&0xFFF;
    if (rb && (st || (m16&1))) { m16++; if(m16>=0x400){m16=0;e16++;} }
    if (e16 >= 31) { uint32_t r=(s<<31)|(0xFF<<23); float f; memcpy(&f,&r,4); return f; }
    uint16_t h = (s<<15)|(e16<<10)|m16;
    uint32_t s2=(uint32_t)((h>>15)&1)<<31, e2=(h>>10)&0x1F, m2=h&0x3FF;
    uint32_t r;
    if (e2==0) r=s2|(m2<<13);
    else if (e2==31) r=s2|(0xFF<<23)|(m2<<13);
    else r=s2|((e2-15+127)<<23)|(m2<<13);
    float f; memcpy(&f,&r,4); return f;
}

static double cast(double x, int p) {
    if (p==16) return fp16((float)x);
    if (p==32) return (float)x;
    return x;
}

static std::vector<std::string> tokenise(const std::string& s) {
    std::vector<std::string> t;
    for (size_t i=0; i<s.size();) {
        if (s[i]==' '||s[i]=='\t'||s[i]=='\n'||s[i]=='\r'){i++;continue;}
        if (s[i]=='('||s[i]==')'){t.push_back(std::string(1,s[i]));i++;continue;}
        size_t j=i; while(j<s.size()&&s[j]!=' '&&s[j]!='('&&s[j]!=')') j++;
        t.push_back(s.substr(i,j-i)); i=j;
    }
    return t;
}

struct Node {
    bool leaf; int idx, prec;
    std::vector<Node*> ch;
    ~Node(){for(auto c:ch)delete c;}
};

static Node* parse(const std::vector<std::string>& t, size_t& i, int n) {
    if (i>=t.size()) throw std::runtime_error("unexpected end");
    if (t[i]=="(") {
        i++;
        std::string p=t[i++];
        int prec = p=="fp16"?16:p=="fp32"?32:p=="fp64"?64:-1;
        if (prec<0) throw std::runtime_error("bad precision: "+p);
        Node* nd=new Node{false,0,prec,{}};
        while(i<t.size()&&t[i]!=")") nd->ch.push_back(parse(t,i,n));
        if(i>=t.size()) throw std::runtime_error("missing )");
        i++;
        if(nd->ch.size()<2) throw std::runtime_error("need >=2 children");
        return nd;
    }
    int idx=std::stoi(t[i++]);
    if(idx<1||idx>n) throw std::runtime_error("index out of range");
    return new Node{true,idx-1,0,{}};
}

static void leaves(Node* nd, std::vector<int>& v) {
    if(nd->leaf){v.push_back(nd->idx);return;}
    for(auto c:nd->ch) leaves(c,v);
}

static void validate(Node* root, int n) {
    std::vector<int> v; leaves(root,v);
    if((int)v.size()!=n) throw std::runtime_error("wrong leaf count");
    std::sort(v.begin(),v.end());
    for(int i=0;i<n;i++) if(v[i]!=i) throw std::runtime_error("indices not unique");
}

static double eval(Node* nd, const std::vector<double>& vals) {
    if(nd->leaf) return vals[nd->idx];
    double acc=cast(eval(nd->ch[0],vals),nd->prec);
    for(size_t i=1;i<nd->ch.size();i++) acc=cast(acc+cast(eval(nd->ch[i],vals),nd->prec),nd->prec);
    return acc;
}

struct Counts { long long a16,a32,a64; };
static Counts count(Node* nd) {
    if(nd->leaf) return {0,0,0};
    Counts c={0,0,0};
    int adds=(int)nd->ch.size()-1;
    if(nd->prec==16) c.a16+=adds;
    else if(nd->prec==32) c.a32+=adds;
    else c.a64+=adds;
    for(auto ch:nd->ch){auto s=count(ch);c.a16+=s.a16;c.a32+=s.a32;c.a64+=s.a64;}
    return c;
}

int main(int argc, char* argv[]) {
    if(argc!=3){std::cerr<<"usage: judge <name> <tests_dir>\n";return 2;}
    std::string dir=argv[2], name=argv[1];
    std::ifstream fin(dir+"/"+name+".in");
    if(!fin){std::cerr<<"cannot open .in\n";return 2;}
    int n; fin>>n;
    std::vector<double> vals(n);
    for(int i=0;i<n;i++) fin>>vals[i];
    fin.close();

    std::ifstream fout(dir+"/"+name+".out");
    if(!fout){std::cerr<<"cannot open .out\n";return 2;}
    std::string sigma_str; fout>>sigma_str;
    fout.close();
    long double sigma=std::stold(sigma_str);

    std::string sched; std::getline(std::cin,sched);

    Node* root=nullptr;
    try {
        auto toks=tokenise(sched);
        size_t pos=0;
        root=parse(toks,pos,n);
        if(pos!=toks.size()) throw std::runtime_error("trailing tokens");
        validate(root,n);
    } catch(std::exception& e) {
        std::cerr<<"INVALID: "<<e.what()<<"\n";
        delete root; return 1;
    }

    double S=eval(root,vals);
    Counts c=count(root);
    delete root;

    const double TAU=1e-10;
    long double eta=fabsl((long double)S-sigma)/(sigma+TAU);
    double alpha = eta==0.0L ? 1.0 : std::min(1.0,std::max(0.0,-(double)(logl(eta)/logl(2.0L))/24.0));
    long long C=c.a16+2*c.a32+8*c.a64;
    double beta=C>0?(double)(n-1)/C:0.0;

    std::cout<<std::fixed<<std::setprecision(10)
             <<100.0*alpha*beta<<" "<<alpha<<" "<<beta<<" "
             <<std::scientific<<(double)eta<<" "<<std::fixed<<C<<"\n";
    return 0;
}
