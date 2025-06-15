#include <bits/stdc++.h>
using namespace std;

// Fast I/O
#define FAST_IO ios_base::sync_with_stdio(0); cin.tie(0); cout.tie(0);

// Constants
#define INF 1e9
#define MOD 1000000007
//#define PI 3.14159265358979323846
#define EPS 1e-9

// Macros for convenience
#define FOR(i, a, b) for (int i = a; i < b; i++)
#define RFOR(i, a, b) for (int i = a; i >= b; i--)
#define ALL(x) x.begin(), x.end()
#define PB push_back
#define MP make_pair
#define F first
#define S second
#define SZ(x) (int)(x.size())

// Debugging helpers
#ifndef ONLINE_JUDGE
#define debug(x) cerr << #x << " = " << x << endl;
#define debug_arr(arr, n) { cerr << #arr << " = "; for (int i = 0; i < n; i++) cerr << arr[i] << " "; cerr << endl; }
#else
#define debug(x) 
#define debug_arr(arr, n)
#endif

// Precomputed values for common cases
vector<int> primes;
void sieve(int limit) {
    vector<bool> is_prime(limit + 1, true);
    is_prime[0] = is_prime[1] = false;
    FOR(i, 2, limit + 1) {
        if (is_prime[i]) {
            primes.PB(i);
            for (int j = i * i; j <= limit; j += i) is_prime[j] = false;
        }
    }
}

// Common Functions
template<typename T> T gcd(T a, T b) { return b == 0 ? a : gcd(b, a % b); }
template<typename T>T lcm(T a, T b) { return a * b / gcd(a, b); }
long long mod_exp(long long base, long long exp, long long mod) {
    long long result = 1;
    while (exp > 0) {
        if (exp % 2 == 1) result = (result * base) % mod;
        base = (base * base) % mod;
        exp /= 2;
    }
    return result;
}
long long mod_inverse(long long x, long long mod) {
    return mod_exp(x, mod - 2, mod);
}

// Fast Input/Output
template <typename T> inline void read(T &x) { cin >> x; }
template <typename T> inline void write(const T &x) { cout << x << endl; }

template <typename T> inline void read_vector(vector<T> &v, int n) {
    v.resize(n);
    FOR(i, 0, n) cin >> v[i];
}

template <typename T> inline void write_vector(const vector<T> &v) {
    for (auto &el : v) cout << el << " ";
    cout << endl;
}

// Data Structures
struct UnionFind {
    vector<int> parent, rank;
    UnionFind(int n) {
        parent.resize(n);
        rank.resize(n, 0);
        iota(parent.begin(), parent.end(), 0);
    }
    int find(int x) {
        if (parent[x] != x) parent[x] = find(parent[x]);
        return parent[x];
    }
    void unite(int x, int y) {
        int rootX = find(x), rootY = find(y);
        if (rootX != rootY) {
            if (rank[rootX] > rank[rootY]) parent[rootY] = rootX;
            else if (rank[rootX] < rank[rootY]) parent[rootX] = rootY;
            else {
                parent[rootY] = rootX;
                rank[rootX]++;
            }
        }
    }
};

// Graph Functions
void bfs(int start, vector<vector<int>>& adj, vector<int>& dist) {
    queue<int> q;
    q.push(start);
    dist[start] = 0;
    while (!q.empty()) {
        int node = q.front(); q.pop();
        for (int neighbor : adj[node]) {
            if (dist[neighbor] == -1) {  // not visited
                dist[neighbor] = dist[node] + 1;
                q.push(neighbor);
            }
        }
    }
}

void dfs(int node, vector<vector<int>>& adj, vector<bool>& visited) {
    visited[node] = true;
    for (int neighbor : adj[node]) {
        if (!visited[neighbor]) dfs(neighbor, adj, visited);
    }
}

int shortest_path_bfs(int start, int target, vector<vector<int>>& adj) {
    vector<int> dist(adj.size(), -1);  // Distance array, initialize to -1
    bfs(start, adj, dist);
    return dist[target];
}

// String Functions
bool is_palindrome(const string &s) {
    int n = s.size();
    FOR(i, 0, n / 2) {
        if (s[i] != s[n - i - 1]) return false;
    }
    return true;
}

string reverse_string(const string &s) {
    string reversed = s;
    reverse(ALL(reversed));
    return reversed;
}

// Sorting and Searching
template <typename T>
void custom_sort(vector<T>& arr) {
    sort(ALL(arr));  // Default sorting (ascending)
}

template <typename T>
int binary_search(const vector<T>& arr, T key) {
    int low = 0, high = arr.size() - 1;
    while (low <= high) {
        int mid = low + (high - low) / 2;
        if (arr[mid] == key) return mid;
        if (arr[mid] < key) low = mid + 1;
        else high = mid - 1;
    }
    return -1;  // Key not found
}

// Modular Arithmetic
long long mod_add(long long a, long long b, long long mod) {
    return (a + b) % mod;
}

long long mod_sub(long long a, long long b, long long mod) {
    return (a - b + mod) % mod;
}

long long mod_mult(long long a, long long b, long long mod) {
    return (a * b) % mod;
}

long long mod_div(long long a, long long b, long long mod) {
    return mod_mult(a, mod_inverse(b, mod), mod);
}

// Geometry Functions
// Cross product of vectors (Ax, Ay) and (Bx, By)
int cross_product(int Ax, int Ay, int Bx, int By) {
    return Ax * By - Ay * Bx;
}

// Area of triangle formed by 3 points (Ax, Ay), (Bx, By), (Cx, Cy)
int area_of_triangle(int Ax, int Ay, int Bx, int By, int Cx, int Cy) {
    return abs(cross_product(Bx - Ax, By - Ay, Cx - Ax, Cy - Ay));
}

// Point Inside Polygon using ray-casting
bool is_point_inside_polygon(int px, int py, const vector<pair<int, int>>& polygon) {
    int n = polygon.size();
    bool inside = false;
    for (int i = 0, j = n - 1; i < n; j = i++) {
        int xi = polygon[i].first, yi = polygon[i].second;
        int xj = polygon[j].first, yj = polygon[j].second;
        
        if ((yi > py) != (yj > py) && px < (xj - xi) * (py - yi) / (yj - yi) + xi) {
            inside = !inside;
        }
    }
    return inside;
}

// Euclidean Distance
double euclidean_distance(int x1, int y1, int x2, int y2) {
    return sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2));
}

// Number Theory Functions
// Prime Factorization
vector<int> prime_factors(int n) {
    vector<int> factors;
    for (int i = 2; i * i <= n; i++) {
        while (n % i == 0) {
            factors.push_back(i);
            n /= i;
        }
    }
    if (n > 1) factors.push_back(n);
    return factors;
}

// Sum of Divisors
int sum_of_divisors(int n) {
    int sum = 1;
    for (int i = 2; i * i <= n; i++) {
        if (n % i == 0) {
            sum += i;
            if (i != n / i) sum += n / i;
        }
    }
    return sum;
}

// Check if a number is a perfect square
bool is_perfect_square(int x) {
    int s = sqrt(x);
    return s * s == x;
}

// Graph Algorithms
// Topological Sort using DFS
bool topological_sort(int n, vector<vector<int>>& adj, vector<int>& result) {
    vector<bool> visited(n, false), inStack(n, false);
    result.clear();

    function<bool(int)> dfs = [&](int u) {
        if (inStack[u]) return false;  // Cycle detected
        if (visited[u]) return true;
        visited[u] = true;
        inStack[u] = true;
        for (int v : adj[u]) {
            if (!dfs(v)) return false;
        }
        inStack[u] = false;
        result.push_back(u);
        return true;
    };

    for (int i = 0; i < n; i++) {
        if (!visited[i]) {
            if (!dfs(i)) return false;
        }
    }
    reverse(result.begin(), result.end());
    return true;
}

// Dijkstra's Algorithm for Single Source Shortest Path
vector<int> dijkstra(int n, int source, vector<vector<pair<int, int>>>& adj) {
    vector<int> dist(n, INT_MAX);
    dist[source] = 0;
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;
    pq.push({0, source});

    while (!pq.empty()) {
        int d = pq.top().first, u = pq.top().second;
        pq.pop();
        if (d > dist[u]) continue;
        for (auto& edge : adj[u]) {
            int v = edge.first, weight = edge.second;
            if (dist[u] + weight < dist[v]) {
                dist[v] = dist[u] + weight;
                pq.push({dist[v], v});
            }
        }
    }
    return dist;
}

// Floyd-Warshall Algorithm for All-Pairs Shortest Path
vector<vector<int>> floyd_warshall(int n, const vector<vector<int>>& adj) {
    vector<vector<int>> dist = adj;  // Copy the adjacency matrix
    for (int k = 0; k < n; k++) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (dist[i][j] > dist[i][k] + dist[k][j]) {
                    dist[i][j] = dist[i][k] + dist[k][j];
                }
            }
        }
    }
    return dist;
}

// String Functions
// Knuth-Morris-Pratt (KMP) Pattern Matching
vector<int> KMP_preprocess(const string& pattern) {
    int m = pattern.size();
    vector<int> lps(m, 0);
    for (int i = 1, len = 0; i < m;) {
        if (pattern[i] == pattern[len]) {
            len++;
            lps[i] = len;
            i++;
        } else {
            if (len != 0) {
                len = lps[len - 1];
            } else {
                lps[i] = 0;
                i++;
            }
        }
    }
    return lps;
}

vector<int> KMP_search(const string& text, const string& pattern) {
    vector<int> result;
    vector<int> lps = KMP_preprocess(pattern);
    int n = text.size(), m = pattern.size();
    for (int i = 0, j = 0; i < n;) {
        if (pattern[j] == text[i]) {
            i++;
            j++;
        }
        if (j == m) {
            result.push_back(i - j);
            j = lps[j - 1];
        } else if (i < n && pattern[j] != text[i]) {
            if (j != 0) {
                j = lps[j - 1];
            } else {
                i++;
            }
        }
    }
    return result;
}

// Miscellaneous Functions
// Find the Kth largest element in an array
int find_kth_largest(vector<int>& arr, int k) {
    nth_element(arr.begin(), arr.begin() + k - 1, arr.end(), greater<int>());
    return arr[k - 1];
}

// Count the number of set bits (1s) in a number
int count_set_bits(int n) {
    int count = 0;
    while (n) {
        n = n & (n - 1); // Drop the lowest set bit
        count++;
    }
    return count;
}
struct Fenw {
    int n;
    vector<int> fenw;
    Fenw(int n): n(n) {
        fenw.assign(n+1, 0);
    }
    // Adds delta to index i.
    void update(int i, int delta) {
        for(; i <= n; i += i & -i)
            fenw[i] += delta;
    }
    // Returns the sum of values from 1 to i.
    int query(int i) {
        int sum = 0;
        for(; i > 0; i -= i & -i)
            sum += fenw[i];
        return sum;
    }
    // Finds the smallest index i such that the prefix sum is at least k.
    int findKth(int k) {
        int idx = 0;
        // Find the highest power of 2 less than or equal to n.
        int bitMask = 1 << (31 - __builtin_clz(n));
        for(; bitMask; bitMask >>= 1) {
            int next = idx + bitMask;
            if(next <= n && fenw[next] < k) {
                k -= fenw[next];
                idx = next;
            }
        }
        return idx + 1;
    }
};
typedef complex<double> cd;
const double PI = acos(-1);

void fft(vector<cd>& a, bool invert) {
    int n = a.size();
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;
        if (i < j)
            swap(a[i], a[j]);
    }
 
    for (int len = 2; len <= n; len <<= 1) {
        double ang = 2 * PI / len * (invert ? -1 : 1);
        cd wlen(cos(ang), sin(ang));
        for (int i = 0; i < n; i += len) {
            cd w(1);
            for (int j = 0; j < len / 2; j++) {
                cd u = a[i + j], v = a[i + j + len / 2] * w;
                a[i + j] = u + v;
                a[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
    if (invert) {
        for (auto & x : a)
            x /= n;
    }
}
bool f(vector<long long> v) {
    while(v.size() > 1) {
        bool ok = true;
        for (int i = 1; i < v.size(); i++) {
            if(v[i-1] >= v[i]) { ok = false; break; }
        }
        if(!ok) return false;
        vector<long long> u;
        for (int i = 1; i < v.size(); i++) {
            u.push_back(v[i] - v[i-1]);
        }
        v = u;
    }
    return true;
}
// Main Function to Solve Test Cases
void solve() {
    // Insert your problem-specific logic here
    long long n;cin>>n;
    for(long long i=1;i<=n;i++){
    	long long res=((i*i)*((i*i)-1)/2)-4*(i-1)*(i-2);
    	cout<<res<<"\n";
	}
}
int main() {
    FAST_IO;  // Enable fast I/O
    
    int t=1;
    //cin >> t;  // Read number of test cases
    while (t--) {
        solve();
    }
    return 0;
}

