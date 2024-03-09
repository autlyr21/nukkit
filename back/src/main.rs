use std::{net::SocketAddr, sync::Arc, thread::spawn};

use axum::{
    body::Body,
    extract::Host,
    handler::HandlerWithoutStateExt,
    http::{header, HeaderMap, Request, Response, StatusCode, Uri},
    response::{IntoResponse, Redirect},
    routing::get,
    Router,
};
use futures::stream::StreamExt;
use lazy_static::lazy_static;
use rustls_acme::{
    axum::AxumAcceptor, caches::DirCache, futures_rustls::rustls::ServerConfig, AcmeConfig,
};
use serde::{Deserialize, Serialize};
use tower::{Layer, ServiceBuilder, ServiceExt};
use tower_http::services::ServeDir;

#[derive(Serialize, Deserialize, Clone, Debug)]
struct Args {
    front_dir: String,
    onnx_dir: String,
    port: u32,
    email: String,
    domains: Vec<String>,
}
lazy_static! {
    static ref STATE: Args = serde_json::from_slice(&std::fs::read("cfg.json").unwrap()).unwrap();
}

#[tokio::main]
async fn main() {
    println!(
        "Front directory: {}\nONNX directory: {}\nPort {}...",
        STATE.front_dir, STATE.onnx_dir, STATE.port
    );

    check_directories(&STATE.front_dir).await;
    check_directories(&STATE.onnx_dir).await;
    let app = Router::new()
        .nest_service("/models/", ServeDir::new(&STATE.onnx_dir))
        .nest_service("/", get(handler));

    let acceptor = create_acceptor().await;
    let _ = spawn_server(app, acceptor).await;
}
async fn spawn_server(app: Router, acceptor: Option<AxumAcceptor>) -> Result<(), std::io::Error> {
    let server = axum_server::bind(
        format!("0.0.0.0:{}", STATE.port)
            .parse()
            .expect("Invalide Port Configuration."),
    );
    match acceptor {
        Some(acceptor) => {
            let _redirect_handle = tokio::spawn(async move { spawn_http_redirect().await });
            server
                .acceptor(acceptor)
                .serve(app.into_make_service())
                .await
        }
        None => server.serve(app.into_make_service()).await,
    }
}
async fn spawn_http_redirect() {
    let addr = SocketAddr::from(([0, 0, 0, 0], 80));
    println!("Binded to port 80.");
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("Failed to bind to port 80.");
    let _ = axum::serve(listener, redirect_http.into_make_service()).await;
}
async fn redirect_http(Host(host): Host, uri: Uri) -> impl IntoResponse {
    let mut parts = uri.into_parts();
    parts.scheme = Some(axum::http::uri::Scheme::HTTPS);
    if parts.path_and_query.is_none() {
        parts.path_and_query = Some("/".parse().unwrap());
    }
    let https_host = host.replace("80", "443");
    parts.authority = Some(https_host.parse().unwrap());
    let uri = Uri::from_parts(parts).unwrap();
    Redirect::permanent(&uri.to_string()).into_response()
}
async fn create_acceptor() -> Option<AxumAcceptor> {
    match STATE.port {
        443 => {
            let mut acme_state = AcmeConfig::new(STATE.domains.clone())
                .contact([format!("mailto:{}", STATE.email)].iter())
                .cache_option(Some(DirCache::new("./cache")))
                .directory_lets_encrypt(true)
                .state();

            let mut rustls_config = ServerConfig::builder()
                .with_no_client_auth()
                .with_cert_resolver(acme_state.resolver());
            rustls_config.alpn_protocols = vec![b"h2".to_vec()];
            let acceptor = acme_state.axum_acceptor(Arc::new(rustls_config));
            tokio::spawn(async move {
                loop {
                    match acme_state.next().await.unwrap() {
                        Ok(ok) => println!("{:?}", ok),
                        Err(err) => println!("{:?}", err),
                    }
                }
            });
            Some(acceptor)
        }
        _ => None,
    }
}

async fn check_directories(dir: &String) {
    match tokio::fs::read_dir(dir).await {
        Ok(_) => {}
        Err(err) => panic!(
            "Failed to read directory {} due to {:?}!",
            STATE.front_dir, err
        ),
    };
}

async fn handler(uri: Uri) -> Result<(HeaderMap, Response<Body>), (StatusCode, String)> {
    let mut headers = HeaderMap::new();
    headers.insert(header::CACHE_CONTROL, "no-store".parse().unwrap());
    headers.insert("Cross-Origin-Opener-Policy", "same-origin".parse().unwrap());
    headers.insert(
        "Cross-Origin-Embedder-Policy",
        "require-corp".parse().unwrap(),
    );
    let res = get_static_file(uri.clone(), headers.clone()).await?;

    if res.1.status() == StatusCode::NOT_FOUND {
        match format!("{}.html", uri).parse() {
            Ok(uri_html) => get_static_file(uri_html, headers).await,
            Err(_) => Err((StatusCode::INTERNAL_SERVER_ERROR, "Invalid URI".to_string())),
        }
    } else {
        Ok(res)
    }
}

async fn get_static_file(
    uri: Uri,
    headers: HeaderMap,
) -> Result<(HeaderMap, Response<Body>), (StatusCode, String)> {
    let req = Request::builder().uri(uri).body(Body::empty()).unwrap();
    match ServeDir::new(&STATE.front_dir).oneshot(req).await {
        Ok(res) => Ok((headers, res.map(Body::new))),
        Err(err) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Something went wrong: {}", err),
        )),
    }
}
