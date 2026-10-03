module DelayConnection

using Parameters: @unpack

mutable struct Delay_Connection{FT}
    N::UInt32               # ニューロンの数
    delay::FT               # 遅延時間
    total_time::FT          # 全体の時間
    dt::FT                  # 時間刻み
    nt::UInt32              # 時間のステップ数
    nt_delay::UInt32        # 遅延のステップ数
    state::Matrix{Bool}     # 状態行列
    out::Vector{Bool}       # 出力ベクトル

    # カスタムコンストラクタ
    function Delay_Connection(N::Integer, delay::FT, total_time::FT, dt::FT) where FT
        nt = UInt32(round(total_time / dt))
        nt_delay = UInt32(round(delay / dt))
        state = falses(N, nt)  # Bool型の初期値はすべて false
        out = falses(N)        # 出力ベクトルも同様に初期化
        new{FT}(UInt32(N), delay, total_time, dt, nt, nt_delay, state, out)
    end
end

# 状態を更新する関数
function Delay_Connection_call!(conn::Delay_Connection, x::Vector{Bool}, pre_t::UInt32)
    exc_delay = conn.nt_delay  # 遅延ステップ数
    if (pre_t + exc_delay) < conn.nt
        conn.state[:, pre_t + exc_delay] = x
    end
    return conn.state[:, pre_t]  # 現時刻の状態を返す
end

mutable struct Delay_Inh_Connection{FT}
    N::UInt32               # ニューロンの数
    delay::FT               # 遅延時間
    total_time::FT          # 全体の時間
    dt::FT                  # 時間刻み
    nt::UInt32              # 時間のステップ数
    nt_delay::UInt32        # 遅延のステップ数
    state::Matrix{Bool}     # 状態行列
    out::Vector{Bool}       # 出力ベクトル

    # カスタムコンストラクタ
    function Delay_Inh_Connection(N::Integer, delay::FT, total_time::FT, dt::FT) where FT
        nt = UInt32(round(total_time / dt))
        nt_delay = UInt32(round(delay / dt))
        state = falses(N, nt)  # Bool型の初期値はすべて false
        out = falses(N)        # 出力ベクトルも同様に初期化
        new{FT}(UInt32(N), delay, total_time, dt, nt, nt_delay, state, out)
    end
end

# 状態を更新する関数
function Delay_Inh_Connection_call!(conn::Delay_Inh_Connection, x::Vector{Bool}, pre_t::UInt32)
    inh_delay = round(Int32, exp(-1* pre_t * conn.dt /500) + conn.nt_delay)
    # inh_delay = round(Int32, conn.nt_delay + (inh_delay/conn.dt))
    if (pre_t + inh_delay) < conn.nt
        conn.state[:, pre_t + inh_delay] = x
    end
    return conn.state[:, pre_t]
end

end