#pragma once

#include "strata/hpu/layer.hpp"
#include "strata/hpu/expert_cache.hpp"
#include <algorithm>
#include <string_view>
#include "strata/hpu/session_flow.hpp"
#include <cerrno>
#include <cstdio>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

namespace strata::hpu {

// Synapse owns one card per process. The second stage exchanges one host residual
// per token and keeps its layer states and weights on its own PCI device.
class PipelineStage {
public:
    struct Resources {
        std::unique_ptr<Device> device;
        std::unique_ptr<ExpertCache> cache;
        std::vector<std::unique_ptr<Layer>> layers;
    };
private:
    enum class Command : uint8_t { checkpoint, decode, restore, reset };
    int socket_=-1;
    pid_t pid_=-1;
    size_t width_;

    static void send_bytes(int fd, const void* data, size_t size) {
        auto* position=static_cast<const char*>(data);
        while(size) {
            const auto written=::send(fd,position,size,MSG_NOSIGNAL);
            if(written<0 && errno==EINTR) continue;
            if(written<=0) throw SessionFatal("Gaudi2 pipeline: worker connection write failed");
            position+=written;size-=written;
        }
    }
    static void receive_bytes(int fd, void* data, size_t size) {
        auto* position=static_cast<char*>(data);
        while(size) {
            const auto received=::recv(fd,position,size,0);
            if(received<0 && errno==EINTR) continue;
            if(received<=0) throw SessionFatal("Gaudi2 pipeline: worker connection closed");
            position+=received;size-=received;
        }
    }
    static void send_status(int fd, std::string_view message={}) {
        const uint32_t length=std::min<size_t>(message.size(),4096);
        send_bytes(fd,&length,sizeof(length));
        send_bytes(fd,message.data(),length);
    }
    void receive_status() {
        uint32_t length;
        receive_bytes(socket_,&length,sizeof(length));
        if(length>4096) throw SessionFatal("Gaudi2 pipeline: invalid worker response");
        if(length) {
            std::string message(length,'\0');
            receive_bytes(socket_,message.data(),message.size());
            throw SessionFatal("Gaudi2 pipeline: "+message);
        }
    }
    void command(Command value) {
        send_bytes(socket_,&value,sizeof(value));
        receive_status();
    }
    void close_worker() noexcept {
        if(socket_>=0) ::close(socket_);
        if(pid_>0) {
            while(::waitpid(pid_,nullptr,0)<0 && errno==EINTR) {}
        }
    }
    static void serve(int fd, size_t width, const std::function<Resources()>& load) {
        auto resources=load();
        auto& layers=resources.layers;
        std::vector<Layer::Checkpoint> checkpoints;
        std::vector<float> residual(width);
        send_status(fd);
        for(;;) {
            Command cmd;
            try {receive_bytes(fd,&cmd,sizeof(cmd));}
            catch(const SessionFatal&) {return;}
            try {
                switch(cmd) {
                    case Command::checkpoint:
                        checkpoints.clear();
                        for(const auto& layer:layers) checkpoints.push_back(layer->checkpoint());
                        break;
                    case Command::decode:
                        if(checkpoints.size()!=layers.size()) throw std::logic_error("missing layer checkpoint");
                        receive_bytes(fd,residual.data(),width*sizeof(float));
                        for(auto& layer:layers) residual=layer->decode(residual,{});
                        if(residual.size()!=width) throw std::logic_error("invalid residual shape");
                        break;
                    case Command::restore:
                        for(size_t i=checkpoints.size();i>0;--i) layers[i-1]->restore(checkpoints[i-1]);
                        checkpoints.clear();
                        break;
                    case Command::reset:
                        for(auto& layer:layers) layer->reset();
                        checkpoints.clear();
                        break;
                    default: throw std::logic_error("unknown pipeline command");
                }
            } catch(const std::exception& error) {
                send_status(fd,error.what());
                continue;
            }
            send_status(fd);
            if(cmd==Command::decode) send_bytes(fd,residual.data(),width*sizeof(float));
        }
    }
public:
    PipelineStage(size_t width, const std::function<Resources()>& load):width_(width) {
        int sockets[2];
        if(::socketpair(AF_UNIX,SOCK_STREAM|SOCK_CLOEXEC,0,sockets)!=0)
            throw std::runtime_error("Gaudi2 pipeline: socketpair failed");
        pid_=::fork();
        if(pid_==0) {
            ::close(sockets[0]);
            int result=0;
            try {serve(sockets[1],width,load);}
            catch(const std::exception& error) {
                try {send_status(sockets[1],error.what());} catch(...) {}
                result=1;
            }
            ::close(sockets[1]);
            ::_exit(result);
        }
        ::close(sockets[1]);
        socket_=sockets[0];
        try {
            if(pid_<0) throw std::runtime_error("Gaudi2 pipeline: fork failed");
            receive_status();
        } catch(...) {close_worker();throw;}
    }
    PipelineStage(const PipelineStage&)=delete;
    PipelineStage& operator=(const PipelineStage&)=delete;
    ~PipelineStage() {close_worker();}

    void checkpoint() {command(Command::checkpoint);}
    void restore() {command(Command::restore);}
    void reset() {command(Command::reset);}
    std::vector<float> decode(std::span<const float> input) {
        if(input.size()!=width_) throw std::logic_error("Gaudi2 pipeline: invalid input shape");
        std::vector<float> output(width_);
        const auto cmd=Command::decode;
        send_bytes(socket_,&cmd,sizeof(cmd));
        send_bytes(socket_,input.data(),input.size_bytes());
        receive_status();
        receive_bytes(socket_,output.data(),output.size()*sizeof(float));
        return output;
    }
};

} // namespace strata::hpu
