// scenarios/financial_crime/green_agent/sidecar/src/main.rs

use axum::{
    body::Body,
    extract::State,
    http::{Request, Response, StatusCode, Uri},
    routing::any,
    Router,
};
use governor::{Quota, RateLimiter};
use hyper::Client;
use std::{net::SocketAddr, sync::Arc, time::Duration};
use tower::ServiceBuilder;
use tower_http::trace::TraceLayer;
use tracing::{error, info, warn};

/// Configuration for the sidecar proxy
#[derive(Clone)]
struct ProxyConfig {
    backend_url: String,
    rate_limiter: Arc<RateLimiter<governor::state::direct::NotKeyed, governor::clock::DefaultClock>>,
    max_retries: u32,
}

impl ProxyConfig {
    fn new() -> Self {
        // Rate limit: 100 requests per second
        let quota = Quota::per_second(std::num::NonZeroU32::new(100).unwrap());
        let rate_limiter = Arc::new(RateLimiter::direct(quota));

        Self {
            backend_url: "http://127.0.0.1:5000".to_string(),
            rate_limiter,
            max_retries: 3,
        }
    }
}

#[tokio::main]
async fn main() {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .init();

    info!("🚀 Green Agent Sidecar Starting...");
    info!("   External: 0.0.0.0:8000");
    info!("   Backend:  127.0.0.1:5000");
    info!("   Rate Limit: 100 req/s");
    info!("   Retries: 3 with jitter");

    let config = ProxyConfig::new();

    // Build the router
    let app = Router::new()
        .fallback(proxy_handler)
        .layer(
            ServiceBuilder::new()
                .layer(TraceLayer::new_for_http())
        )
        .with_state(config);

    // Bind to 0.0.0.0:8000 (external)
    let addr = SocketAddr::from(([0, 0, 0, 0], 8000));
    
    info!("✅ Sidecar listening on {}", addr);
    
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("Failed to bind to address");

    axum::serve(listener, app)
        .await
        .expect("Server failed");
}

/// Main proxy handler with rate limiting and retries
async fn proxy_handler(
    State(config): State<ProxyConfig>,
    mut req: Request<Body>,
) -> Result<Response<Body>, StatusCode> {
    // Check rate limit
    if config.rate_limiter.check().is_err() {
        warn!("⚠️  Rate limit exceeded");
        return Err(StatusCode::TOO_MANY_REQUESTS);
    }

    // Build backend URL
    let path = req.uri().path();
    let query = req.uri().query().unwrap_or("");
    let backend_uri = format!(
        "{}{}{}",
        config.backend_url,
        path,
        if query.is_empty() {
            String::new()
        } else {
            format!("?{}", query)
        }
    );

    // Update request URI
    *req.uri_mut() = backend_uri.parse().unwrap();

    // Retry with exponential backoff + jitter
    let mut attempt = 0;
    loop {
        attempt += 1;

        match forward_request(req.clone()).await {
            Ok(response) => {
                if attempt > 1 {
                    info!("✅ Request succeeded on retry {}", attempt);
                }
                return Ok(response);
            }
            Err(e) if attempt < config.max_retries => {
                // Calculate backoff with jitter
                let base_delay = Duration::from_millis(500);
                let exponential_delay = base_delay * 2_u32.pow(attempt - 1);
                let jitter = Duration::from_millis(rand::random::<u64>() % 1000);
                let total_delay = exponential_delay + jitter;

                warn!(
                    "⚠️  Attempt {} failed: {}. Retrying in {:?}...",
                    attempt, e, total_delay
                );

                tokio::time::sleep(total_delay).await;
            }
            Err(e) => {
                error!("❌ All {} attempts failed: {}", config.max_retries, e);
                return Err(StatusCode::BAD_GATEWAY);
            }
        }
    }
}

/// Forward request to backend
async fn forward_request(req: Request<Body>) -> Result<Response<Body>, Box<dyn std::error::Error>> {
    let client = Client::new();
    
    let response = client
        .request(req)
        .await
        .map_err(|e| format!("Backend request failed: {}", e))?;

    Ok(response)
}
